import sys
import subprocess
import os
import signal

# --- BOOTSTRAP DEPENDENCIES ---
def bootstrap():
    required = ["PySide6", "psutil", "pyqtgraph"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Dependencies installed. Restarting application...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)

bootstrap()

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AV1 Encoder Pro")
    app.setOrganizationName("AV1Runner")
    
    win = MainWindow()
    win.show()
    
    # Allow Ctrl+C to kill
    signal.signal(signal.SIGINT, lambda *args: win.close())
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
