import { describe, expect, it } from "vitest";
import {
  AdaptiveAecFilter,
  AudioCapturePipeline,
  computeRmsDb,
  Int16RingBuffer,
  PlaybackQueue,
  PreRollBuffer,
  toMono16kPcm16,
} from "@voice-agent/audio";
import { MetricsCollector } from "@voice-agent/metrics";
import {
  createEvent,
  decodeAudioFrame,
  encodeAudioFrame,
  GenerationTracker,
  parseEventEnvelope,
  StreamKind,
} from "@voice-agent/protocol";

describe("protocol codec and lifecycle guard", () => {
  it("roundtrips 24-byte binary PCM frames and rejects malformed packets", () => {
    const pcm = new Int16Array([0, 1234, -5678, 32767, -32768]);
    const encoded = encodeAudioFrame({
      streamKind: StreamKind.TtsOutput,
      seq: 42,
      ptsMs: 1234.5,
      sampleRate: 24000,
      generationTag: 7,
      pcm16: pcm,
    });

    const decoded = decodeAudioFrame(encoded);
    expect(decoded.streamKind).toBe(StreamKind.TtsOutput);
    expect(decoded.seq).toBe(42);
    expect(decoded.ptsMs).toBeCloseTo(1234.5, 4);
    expect(decoded.sampleRate).toBe(24000);
    expect(decoded.generationTag).toBe(7);
    expect(Array.from(decoded.pcm16)).toEqual(Array.from(pcm));

    // Trust-boundary checks.
    expect(() => decodeAudioFrame(new Uint8Array(10))).toThrow(/too short/);
    const badMagic = encoded.slice();
    badMagic[0] = 0x00;
    expect(() => decodeAudioFrame(badMagic)).toThrow(/magic/);
    expect(() => decodeAudioFrame(new Uint8Array(25))).toThrow(/even/);
  });

  it("isolates stale generation events after cancellation", () => {
    const tracker = new GenerationTracker("sess_1");
    const t1 = tracker.startTurn();
    const g1 = tracker.startGeneration();

    const activeEvt = createEvent("asr.partial", { sessionId: "sess_1", turnId: t1, generationId: g1.generationId }, {
      text: "hello",
      stability: 0.9,
      audio_pts_ms: 100,
      compute_ms: 5,
    });
    expect(tracker.isEventActive(activeEvt)).toBe(true);
    expect(tracker.isGenerationTagActive(g1.generationTag)).toBe(true);

    tracker.cancelCurrentGeneration();
    expect(tracker.isEventActive(activeEvt)).toBe(false);
    expect(tracker.isGenerationTagActive(g1.generationTag)).toBe(false);

    const parsed = parseEventEnvelope(JSON.stringify(activeEvt));
    expect(parsed.type).toBe("asr.partial");
  });
});

describe("audio ring buffer, resampler, AEC, and playback queue", () => {
  it("preserves exact newest samples in ring buffer and 300ms pre-roll", () => {
    const ring = new Int16RingBuffer(4);
    ring.push(new Int16Array([1, 2, 3]));
    ring.push(new Int16Array([4, 5, 6]));
    expect(Array.from(ring.readLatest())).toEqual([3, 4, 5, 6]);

    const pre = new PreRollBuffer(10, 1000); // 10 samples
    pre.push(new Int16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    expect(Array.from(pre.snapshot())).toEqual([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it("downmixes 48kHz stereo to 16kHz mono PCM16 accurately", () => {
    const stereoFrames = 480; // 10ms at 48kHz
    const interleaved = new Float32Array(stereoFrames * 2);
    for (let i = 0; i < stereoFrames; i += 1) {
      interleaved[i * 2] = 0.5;
      interleaved[i * 2 + 1] = 0.5;
    }
    const pcm16k = toMono16kPcm16(interleaved, 48000, 2);
    expect(pcm16k.length).toBe(160); // 10ms at 16kHz
    expect(computeRmsDb(pcm16k)).toBeGreaterThan(-7);
  });

  it("attenuates linear speaker echo via adaptive NLMS AEC", () => {
    const aec = new AdaptiveAecFilter(32, 0.35);
    const n = 3200;
    const ref = new Int16Array(n);
    const mic = new Int16Array(n);
    for (let i = 0; i < n; i += 1) {
      ref[i] = Math.round(Math.sin(i * 0.13) * 8000 + Math.cos(i * 0.29) * 4000);
      mic[i] = i >= 6 ? Math.round(ref[i - 6]! * 0.65) : 0;
    }
    const cleaned = aec.process(mic, ref);
    // Measure echo suppression after filter convergence on second half.
    const rawDb = computeRmsDb(mic.subarray(n / 2));
    const cleanDb = computeRmsDb(cleaned.subarray(n / 2));
    expect(rawDb - cleanDb).toBeGreaterThan(8);
  });

  it("emits continuous 20ms frames and tracks playback cursor with Duck/Pause/Resume/Cancel", () => {
    const capture = new AudioCapturePipeline({ frameDurationMs: 20, noiseSuppressionEnabled: false });
    const emitted: number[] = [];
    capture.onFrame((f) => emitted.push(f.pcm16.length));
    capture.ingestPcm16k(new Int16Array(640), 1000);
    expect(emitted).toEqual([320, 320]);

    const playback = new PlaybackQueue();
    const aecRefs: number[] = [];
    playback.onAecReference((r) => aecRefs.push(r.length));

    const chunkSamples = new Int16Array(2400).fill(10000); // 100ms at 24kHz
    playback.enqueueChunk({
      turnId: "turn_0001",
      generationId: "gen_1",
      generationTag: 1,
      sentenceIndex: 0,
      chunkIndex: 0,
      charStart: 0,
      charEnd: 40,
      sampleRate: 24000,
      pcm16: chunkSamples,
    });

    const firstHalf = playback.pullNextAudio(1200);
    expect(firstHalf?.pcm16.length).toBe(1200);
    expect(playback.getCurrentCursor().playedCharIndex).toBe(20);
    expect(aecRefs.length).toBe(1);

    playback.duck(0.2);
    expect(playback.getMode()).toBe("ducked");
    const duckedSlice = playback.pullNextAudio(600);
    expect(duckedSlice?.pcm16[0]).toBe(2000);

    const pausedCursor = playback.pause();
    expect(playback.getMode()).toBe("paused");
    expect(playback.pullNextAudio(600)).toBeNull();
    expect(pausedCursor.playedCharIndex).toBe(30);

    playback.resume();
    expect(playback.getMode()).toBe("playing");

    const cancelCursor = playback.cancel("gen_1");
    expect(cancelCursor.playedCharIndex).toBe(30);
    expect(playback.getMode()).toBe("idle");

    // Stale chunk from cancelled generation must be dropped.
    playback.enqueueChunk({
      turnId: "turn_0001",
      generationId: "gen_1",
      generationTag: 1,
      sentenceIndex: 0,
      chunkIndex: 1,
      charStart: 40,
      charEnd: 80,
      sampleRate: 24000,
      pcm16: chunkSamples,
    });
    expect(playback.getMode()).toBe("idle");
  });

  it("summarizes cold vs hot E2E audible latencies accurately", () => {
    const metrics = new MetricsCollector();
    metrics.startTurn("t1");
    metrics.mark("t1", { speechEndMs: 1000, audiblePlayMs: 2800, vramUsedMb: 5200 });
    metrics.startTurn("t2");
    metrics.mark("t2", { speechEndMs: 5000, audiblePlayMs: 5900, vramUsedMb: 5400 });

    const summary = metrics.summarize();
    expect(summary.coldTurns).toBe(1);
    expect(summary.hotTurns).toBe(1);
    expect(summary.e2eAudibleLatencyMs.p50).toBe(900);
    expect(summary.hotE2eAudibleLatencyMs.p50).toBe(900);
    expect(summary.peakVramMb).toBe(5400);
  });
});
