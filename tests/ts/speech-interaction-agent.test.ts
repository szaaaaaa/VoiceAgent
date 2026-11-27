import path from "node:path";
import { describe, expect, it } from "vitest";
import { PlaybackQueue } from "@voice-agent/audio";
import {
  isDangerousCommand,
  isPathWithinAllowedRoots,
  parseSpokenApproval,
  PermissionGatekeeper,
  PiAgentConnector,
  type LlmStreamProvider,
} from "@voice-agent/agent";
import {
  HybridTurnManager,
  InterruptionController,
  PlaybackAwareHistory,
  SpeculativeGenerationController,
} from "@voice-agent/interaction";
import { MetricsCollector } from "@voice-agent/metrics";
import { GenerationTracker } from "@voice-agent/protocol";
import { sanitizeForSpeech, SentenceBuffer, VadStateMachine } from "@voice-agent/speech";

describe("VAD and speakable TTS text sanitizer / sentence buffer", () => {
  it("applies VAD hysteresis and ignores brief noise spikes", () => {
    const vad = new VadStateMachine({ minSpeechDurationMs: 100, minSilenceDurationMs: 160 });
    expect(vad.update(0.8, 0)).toBeNull();
    expect(vad.update(0.1, 40)).toBeNull(); // brief spike ignored
    expect(vad.update(0.85, 100)).toBeNull();
    expect(vad.update(0.9, 200)?.event).toBe("vad.start");
    expect(vad.update(0.2, 300)).toBeNull();
    expect(vad.update(0.1, 460)?.event).toBe("vad.end");
  });

  it("strips code blocks, <think> reasoning, and raw tool JSON from streaming TTS output", () => {
    const buf = new SentenceBuffer({ firstClauseMinChars: 10, sentenceMinChars: 18 });
    const emitted = [
      ...buf.pushDelta("<think>internal reasoning</think>Sure, I checked your "),
      ...buf.pushDelta("workspace files. ```bash\nrm -rf /tmp/x\n``` Here is "),
      ...buf.pushDelta("the summary you asked for."),
      ...buf.flush(),
    ];

    const joined = emitted.map((s) => s.text).join(" ");
    expect(joined).toContain("Sure, I checked your workspace files.");
    expect(joined).toContain("Here is the summary you asked for.");
    expect(joined).not.toContain("internal reasoning");
    expect(joined).not.toContain("rm -rf");
    expect(sanitizeForSpeech('Result: {"tool":"fs.read","path":"a.txt"} done')).not.toContain("fs.read");
  });
});

describe("hybrid turn manager, full-duplex interruption, and speculative isolation", () => {
  it("classifies EOT, HOLD, Backchannel, Interruption, and Noise accurately", () => {
    const tm = new HybridTurnManager({ strategy: "hybrid" });

    // EOT
    expect(
      tm.evaluate({
        vadInSpeech: false,
        speechDurationMs: 1400,
        silenceDurationMs: 320,
        assistantSpeaking: false,
        transcript: "Open the browser please.",
        adapter: { pCompletion: 0.88, pFloorTaking: 0.05, pNonFloorFeedback: 0.02 },
      }).event,
    ).toBe("EOT");

    // HOLD mid-sentence hesitation
    expect(
      tm.evaluate({
        vadInSpeech: false,
        speechDurationMs: 900,
        silenceDurationMs: 350,
        assistantSpeaking: false,
        transcript: "Help me open um...",
        adapter: { pCompletion: 0.74, pFloorTaking: 0.05, pNonFloorFeedback: 0.02 },
      }).event,
    ).toBe("HOLD");

    // Conservative silence timeout fallback (FR-20)
    expect(
      tm.evaluate({
        vadInSpeech: false,
        speechDurationMs: 900,
        silenceDurationMs: 1200,
        assistantSpeaking: false,
        transcript: "Help me open um",
        adapter: { pCompletion: 0.3, pFloorTaking: 0.05, pNonFloorFeedback: 0.02 },
      }).event,
    ).toBe("EOT");

    // Backchannel during playback
    expect(
      tm.evaluate({
        vadInSpeech: false,
        speechDurationMs: 280,
        silenceDurationMs: 100,
        assistantSpeaking: true,
        transcript: "uh-huh",
        adapter: { pCompletion: 0.1, pFloorTaking: 0.08, pNonFloorFeedback: 0.85 },
      }).event,
    ).toBe("BACKCHANNEL");

    // Active interruption during playback
    expect(
      tm.evaluate({
        vadInSpeech: true,
        speechDurationMs: 420,
        silenceDurationMs: 0,
        assistantSpeaking: true,
        transcript: "Wait, check CPU instead",
        adapter: { pCompletion: 0.1, pFloorTaking: 0.91, pNonFloorFeedback: 0.03 },
      }).event,
    ).toBe("INTERRUPTION");

    // Ambient noise during playback
    expect(
      tm.evaluate({
        vadInSpeech: false,
        speechDurationMs: 90,
        silenceDurationMs: 50,
        assistantSpeaking: true,
        transcript: "",
        adapter: { pCompletion: 0.05, pFloorTaking: 0.08, pNonFloorFeedback: 0.1 },
      }).event,
    ).toBe("NOISE");

    // Scenario 6a: Short interjection yielding floor back ("sorry go on" / "你继续") -> resume
    const yieldDecision = tm.evaluate({
      vadInSpeech: false,
      speechDurationMs: 360,
      silenceDurationMs: 260,
      assistantSpeaking: true,
      transcript: "你继续",
    });
    expect(yieldDecision.event).toBe("SHORT_INTERJECTION");
    expect(yieldDecision.action).toBe("resume_playback");

    // Scenario 6b: Short directive interjection followed by silence -> cancel and regenerate
    const regenDecision = tm.evaluate({
      vadInSpeech: false,
      speechDurationMs: 420,
      silenceDurationMs: 280,
      assistantSpeaking: true,
      transcript: "换成中文回答",
    });
    expect(regenDecision.event).toBe("SHORT_INTERJECTION");
    expect(regenDecision.action).toBe("cancel_and_regenerate");
  });

  it("resumes on backchannel and cancels + preserves executed tool history on interruption", () => {
    const playback = new PlaybackQueue();
    const tracker = new GenerationTracker("sess_fd");
    const metrics = new MetricsCollector();
    const history = new PlaybackAwareHistory();

    const turnId = tracker.startTurn();
    const gen = tracker.startGeneration();
    metrics.startTurn(turnId);

    history.recordUserTurn(turnId, "Check disk usage and summarize");
    history.recordToolExecution(turnId, {
      toolCallId: "tc_1",
      toolName: "system.info",
      argsSummary: "{}",
      status: "ok",
      sideEffectCommitted: true,
      result: "16GB RAM free",
    });
    history.recordToolExecution(turnId, {
      toolCallId: "tc_2",
      toolName: "fs.read",
      argsSummary: '{"path":"notes.md"}',
      status: "running",
      sideEffectCommitted: false,
      result: "in_flight",
    });
    const fullText = "Your system has sixteen gigabytes free and all drives are healthy.";
    history.recordAssistantReply(turnId, fullText, fullText);

    playback.enqueueChunk({
      turnId,
      generationId: gen.generationId,
      generationTag: gen.generationTag,
      sentenceIndex: 0,
      chunkIndex: 0,
      charStart: 0,
      charEnd: fullText.length,
      sampleRate: 24000,
      pcm16: new Int16Array(2400).fill(5000),
    });
    playback.pullNextAudio(1200); // ~half played

    let abortedGen = "";
    const ic = new InterruptionController(playback, tracker, metrics, {
      onAbortGeneration: (genId, _tag, cursor) => {
        abortedGen = genId;
        history.markInterrupted(cursor.turnId, cursor.playedCharIndex);
      },
    });

    // Backchannel -> Duck then Resume without cancel.
    ic.onPotentialUserSpeech();
    expect(playback.getMode()).toBe("ducked");
    const bcRes = ic.resolve({ event: "BACKCHANNEL", confidence: 0.85, source: "test", reason: "ok" });
    expect(bcRes.resolution).toBe("resumed");
    expect(playback.getMode()).toBe("playing");

    // Interruption -> Cancel playback, abort generation, truncate audible history, keep tool result.
    ic.onPotentialUserSpeech();
    const intRes = ic.resolve({ event: "INTERRUPTION", confidence: 0.92, source: "test", reason: "barge-in" });
    expect(intRes.resolution).toBe("cancelled");
    expect(abortedGen).toBe(gen.generationId);
    expect(playback.getMode()).toBe("idle");

    const msgs = history.toAgentMessages();
    expect(msgs.some((m) => m.role === "tool" && m.content.includes("16GB RAM free"))).toBe(true);
    const assistantMsg = msgs.find((m) => m.role === "assistant")!;
    expect(assistantMsg.content).toContain("[interrupted by user");
    expect(assistantMsg.content.length).toBeLessThan(fullText.length + 45);

    const syncSnap = history.getTurnSyncSnapshot(turnId)!;
    expect(syncSnap.heardPrefix.length).toBeGreaterThan(0);
    expect(syncSnap.unheardSuffix.length).toBeGreaterThan(0);
    expect(syncSnap.committedTools.map((t) => t.toolCallId)).toEqual(["tc_1"]);
    expect(syncSnap.cancelledOrDeniedTools[0]?.status).toBe("cancelled");
  });

  it("enforces speculative generation commit/discard and forbids external tools", () => {
    const spec = new SpeculativeGenerationController({ triggerProbability: 0.6, minSimilarity: 0.8 });
    expect(spec.shouldTrigger(0.75, "check gpu usage")).toBe(true);
    const cand = spec.startCandidate("t1", "spec_1", "check gpu usage");
    expect(cand.allowExternalTools).toBe(false);
    spec.bufferDelta("spec_1", "Checking GPU...");

    const commit = spec.confirmFinalTranscript("check gpu usage please");
    expect(commit.hit).toBe(true);
    expect(commit.deltas).toEqual(["Checking GPU..."]);

    const cand2 = spec.startCandidate("t2", "spec_2", "open notepad");
    const miss = spec.confirmFinalTranscript("actually shut down music player");
    expect(miss.hit).toBe(false);
    expect(cand2.abortController.signal.aborted).toBe(true);
  });
});

describe("permission gatekeeper and Pi Agent 20-turn session stability", () => {
  it("blocks path escapes, force-delete commands, and fails closed on ambiguous/timeout approvals", async () => {
    const root = path.resolve("d:/voice-agent");
    expect(isPathWithinAllowedRoots(path.join(root, "docs/README.md"), [root])).toBe(true);
    expect(isPathWithinAllowedRoots(path.join(root, "../outside.txt"), [root])).toBe(false);

    expect(isDangerousCommand("rm -rf /tmp/foo")).toBe(true);
    expect(isDangerousCommand("Remove-Item -Recurse -Force ./build")).toBe(true);
    expect(isDangerousCommand("ri -r -fo ./build")).toBe(true);
    expect(isDangerousCommand("git reset --hard HEAD")).toBe(true);
    expect(isDangerousCommand("git status")).toBe(false);

    expect(parseSpokenApproval("").approved).toBe(false);
    expect(parseSpokenApproval("hmm maybe").approved).toBe(false);
    expect(parseSpokenApproval("嗯？").approved).toBe(false);
    expect(parseSpokenApproval("yes").approved).toBe(true);
    expect(parseSpokenApproval("确认").approved).toBe(true);

    const gate = new PermissionGatekeeper({
      mode: "conversation",
      level: "ask",
      allowedRoots: [root],
      approvalTimeoutMs: 50,
    });

    // Conversation mode blocks write tools.
    const convWrite = await gate.authorize({
      toolName: "fs.write",
      riskLevel: "write",
      targetPath: path.join(root, "tmp_test.txt"),
      impactSummary: "write file",
    });
    expect(convWrite.approved).toBe(false);

    // Action + Ask times out fail-closed if user stays silent.
    gate.setMode("action");
    const timedOut = await gate.authorize(
      {
        toolName: "fs.write",
        riskLevel: "write",
        targetPath: path.join(root, "tmp_test.txt"),
        impactSummary: "write file",
      },
      () => new Promise(() => {}), // never resolves -> must time out as denied
    );
    expect(timedOut.approved).toBe(false);
    expect(timedOut.source).toBe("timeout");
  });

  it("completes 20+ continuous multi-turn interactions with guarded tool execution (Phase 1 gate)", async () => {
    const gate = new PermissionGatekeeper({
      mode: "conversation",
      level: "ask",
      allowedRoots: [path.resolve("d:/voice-agent")],
    });
    const mockProvider: LlmStreamProvider = async function* (req) {
      const lastUser = req.messages[req.messages.length - 1]?.content ?? "";
      if (lastUser.includes("system info")) {
        yield {
          type: "tool_call",
          call: { toolCallId: `call_${req.messages.length}`, toolName: "system.info", args: {} },
        };
      }
      if (lastUser.includes("write secret")) {
        yield {
          type: "tool_call",
          call: {
            toolCallId: `call_write_${req.messages.length}`,
            toolName: "fs.write",
            args: { path: path.resolve("d:/voice-agent/unauthorized.txt"), content: "blocked" },
          },
        };
      }
      yield { type: "delta", text: `Reply for: ${lastUser}.` };
    };

    const agent = new PiAgentConnector(gate, mockProvider);
    for (let i = 1; i <= 20; i += 1) {
      const prompt = i === 1 ? "check system info" : i === 2 ? "write secret file" : `turn number ${i}`;
      const res = await agent.runTurn({
        turnId: `turn_${i}`,
        generationId: `gen_${i}`,
        userTranscript: prompt,
      });
      expect(res.aborted).toBe(false);
      expect(res.speakableReply).toContain(`Reply for: ${prompt}`);
    }

    const turns = agent.history.getTurns();
    expect(turns.length).toBe(20);
    expect(turns[0]!.toolCalls[0]!.status).toBe("ok");
    expect(turns[1]!.toolCalls[0]!.status).toBe("denied");
  });
});
