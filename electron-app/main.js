const { app, BrowserWindow } = require('electron');
const exec = require('child_process').exec;

function createWindow () {
  let win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  });
  
  win.loadURL('http://localhost:8501');
}

app.whenReady().then(() => {
  exec('docker run -p 8501:8501 nutrack:latest', (err, stdout, stderr) => {
    if (err) {
      console.error(`Erreur: ${stderr}`);
      return;
    }
    console.log(`Sortie: ${stdout}`);
  });
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
