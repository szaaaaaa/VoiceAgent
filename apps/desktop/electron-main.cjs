/**
 * Standalone Electron desktop launcher for VoiceAgent GUI.
 * Loads the Vite dev server (or built dist/renderer) in a hardened BrowserWindow
 * with contextIsolation=true, nodeIntegration=false, and localhost mic permission.
 */

const path = require("node:path");
const { spawn } = require("node:child_process");
const { app, BrowserWindow, session } = require("electron");

let backendProc = null;

function startBackendIfNeeded() {
  if (process.env.VOICE_AGENT_SKIP_BACKEND_SPAWN === "1") return;
  const pythonExe = path.resolve(__dirname, "../../.venv/Scripts/python.exe");
  const host = process.env.VOICE_AGENT_WS_HOST || "127.0.0.1";
  const port = process.env.VOICE_AGENT_WS_PORT || "8765";
  backendProc = spawn(
    pythonExe,
    ["-m", "uvicorn", "inference.server:app", "--host", host, "--port", port],
    { cwd: path.resolve(__dirname, "../.."), stdio: "inherit", windowsHide: true },
  );
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1120,
    height: 780,
    backgroundColor: "#0b0f17",
    title: "VoiceAgent",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const devUrl = process.env.VOICE_AGENT_RENDERER_URL || "http://127.0.0.1:5173";
  win.loadURL(devUrl).catch(() => {
    win.loadFile(path.join(__dirname, "dist/renderer/index.html"));
  });
}

app.whenReady().then(() => {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const url = webContents.getURL();
    const isLocal = url.startsWith("http://127.0.0.1") || url.startsWith("file://");
    callback(isLocal && permission === "media");
  });

  startBackendIfNeeded();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (backendProc && !backendProc.killed) backendProc.kill();
  if (process.platform !== "darwin") app.quit();
});
