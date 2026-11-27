/**
 * Standalone Electron desktop main process for VoiceAgent.
 * - Auto-spawns and supervises the local Python speech inference server (127.0.0.1:8765)
 * - Loads built dist/renderer/index.html directly (zero dev server required) or HMR dev URL
 * - Registers Alt+Space global Push-to-Talk shortcut & native JSONL log export dialog
 * - Supports VOICE_AGENT_SMOKE_TEST=1 for automated headless startup verification
 */

const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { app, BrowserWindow, dialog, globalShortcut, ipcMain, session } = require("electron");

const ROOT_DIR = path.resolve(__dirname, "../..");
const HOST = process.env.VOICE_AGENT_WS_HOST || "127.0.0.1";
const PORT = Number(process.env.VOICE_AGENT_WS_PORT || 8765);
const BASE_URL = `http://${HOST}:${PORT}`;

// Isolate Electron userData cache inside the workspace so parallel/repeated launches never lock.
app.setPath("userData", path.join(ROOT_DIR, ".electron-userdata"));

let mainWin = null;
let backendProc = null;
let heavyModelsEnabled = process.env.VOICE_AGENT_LOAD_HEAVY_MODELS === "1";

async function isBackendHealthy() {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

function stopBackend() {
  if (backendProc && !backendProc.killed) {
    backendProc.kill();
    backendProc = null;
  }
}

async function startBackend(loadHeavy = heavyModelsEnabled) {
  heavyModelsEnabled = Boolean(loadHeavy);
  if (process.env.VOICE_AGENT_SKIP_BACKEND_SPAWN === "1") return true;
  if (await isBackendHealthy()) return true;

  const pythonExe = path.join(ROOT_DIR, ".venv", "Scripts", "python.exe");
  const cmd = fs.existsSync(pythonExe) ? pythonExe : "python";
  const hfCache = path.join(ROOT_DIR, "models", "hf_cache");

  backendProc = spawn(
    cmd,
    ["-m", "uvicorn", "inference.server:app", "--host", HOST, "--port", String(PORT)],
    {
      cwd: ROOT_DIR,
      stdio: "inherit",
      windowsHide: true,
      env: {
        ...process.env,
        HF_HOME: process.env.HF_HOME || hfCache,
        VOICE_AGENT_WS_HOST: HOST,
        VOICE_AGENT_WS_PORT: String(PORT),
        VOICE_AGENT_LOAD_HEAVY_MODELS: heavyModelsEnabled ? "1" : "0",
      },
    },
  );

  backendProc.on("exit", () => {
    backendProc = null;
  });

  const deadline = Date.now() + 25000;
  while (Date.now() < deadline) {
    if (await isBackendHealthy()) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

function createWindow() {
  const isSmoke = process.env.VOICE_AGENT_SMOKE_TEST === "1";
  mainWin = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 900,
    minHeight: 640,
    show: !isSmoke,
    backgroundColor: "#0b0f17",
    icon: path.join(__dirname, "assets", "voiceagent.ico"),
    title: "VoiceAgent — 本地全双工语音智能体",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  const devUrl = process.env.VOICE_AGENT_RENDERER_URL;
  const targetUrl = devUrl || `${BASE_URL}/`;

  if (isSmoke) {
    mainWin.webContents.on("console-message", (ev) => {
      if (ev.level >= 2) process.stderr.write(`[renderer-console:${ev.level}] ${ev.message}\n`);
    });
    mainWin.webContents.once("did-fail-load", (_ev, code, desc) => {
      process.stderr.write(`SMOKE_LOAD_FAIL ${code} ${desc}\n`);
      stopBackend();
      app.exit(1);
    });
    mainWin.webContents.once("did-finish-load", async () => {
      try {
        await new Promise((r) => setTimeout(r, 1000));
        const title = await mainWin.webContents.executeJavaScript("document.title");
        const snapshot = await mainWin.webContents.executeJavaScript(
          "({ hasRoot: Boolean(document.getElementById('root') && document.getElementById('root').innerText.includes('VoiceAgent')), hasDesktopBridge: Boolean(window.voiceAgentDesktop && window.voiceAgentDesktop.isElectron), hasHud: Boolean(document.getElementById('root') && document.getElementById('root').innerText.includes('开发者实时监控面板')), textLen: (document.getElementById('root')?.innerText || '').length })",
        );
        const healthy = await isBackendHealthy();
        const report = JSON.stringify({ smoke: "ok", url: targetUrl, title, ...snapshot, backendHealthy: healthy });
        process.stdout.write(`${report}\n`);
        fs.writeFileSync(path.join(ROOT_DIR, ".electron-userdata", "smoke-result.json"), report, "utf8");
        stopBackend();
        app.exit(snapshot.hasRoot && snapshot.hasDesktopBridge && healthy ? 0 : 1);
      } catch (err) {
        process.stderr.write(`SMOKE_FAIL ${String(err)}\n`);
        stopBackend();
        app.exit(1);
      }
    });
  }

  mainWin.loadURL(targetUrl);

  mainWin.on("closed", () => {
    mainWin = null;
  });
}

function registerIpc() {
  ipcMain.handle("desktop:get-backend-config", () => ({
    host: HOST,
    port: PORT,
    wsUrl: `ws://${HOST}:${PORT}/ws/v1/voice`,
    apiBase: BASE_URL,
    heavyModelsEnabled,
  }));

  ipcMain.handle("desktop:restart-backend", async (_ev, opts = {}) => {
    stopBackend();
    await new Promise((r) => setTimeout(r, 300));
    const ok = await startBackend(Boolean(opts.loadHeavy));
    return { ok, heavyModelsEnabled };
  });

  ipcMain.handle("desktop:export-logs", async (_ev, jsonlContent) => {
    if (!mainWin) return { saved: false };
    const res = await dialog.showSaveDialog(mainWin, {
      title: "Export VoiceAgent Experiment Logs",
      defaultPath: path.join(ROOT_DIR, `voiceagent_logs_${Date.now()}.jsonl`),
      filters: [{ name: "JSON Lines", extensions: ["jsonl", "json"] }],
    });
    if (res.canceled || !res.filePath) return { saved: false };
    fs.writeFileSync(res.filePath, String(jsonlContent || ""), "utf8");
    return { saved: true, path: res.filePath };
  });

  ipcMain.handle("desktop:toggle-devtools", () => {
    if (mainWin) mainWin.webContents.toggleDevTools();
  });
}

app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const url = webContents.getURL();
    const isLocal = url.startsWith("http://127.0.0.1") || url.startsWith("file://");
    callback(isLocal && (permission === "media" || permission === "audioCapture"));
  });

  registerIpc();
  await startBackend(heavyModelsEnabled);
  createWindow();

  if (process.env.VOICE_AGENT_SMOKE_TEST !== "1") {
    globalShortcut.register("Alt+Space", () => {
      mainWin?.webContents.send("desktop:ptt-toggle");
    });
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  stopBackend();
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});
