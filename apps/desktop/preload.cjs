/**
 * Electron Preload Script: exposes a minimal, typed desktop API to the Renderer
 * while keeping contextIsolation=true and nodeIntegration=false.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("voiceAgentDesktop", {
  isElectron: true,
  getBackendConfig: () => ipcRenderer.invoke("desktop:get-backend-config"),
  restartBackend: (opts) => ipcRenderer.invoke("desktop:restart-backend", opts),
  exportLogs: (jsonlContent) => ipcRenderer.invoke("desktop:export-logs", jsonlContent),
  toggleDevTools: () => ipcRenderer.invoke("desktop:toggle-devtools"),
  onPushToTalkToggle: (cb) => {
    const handler = () => cb();
    ipcRenderer.on("desktop:ptt-toggle", handler);
    return () => ipcRenderer.removeListener("desktop:ptt-toggle", handler);
  },
});
