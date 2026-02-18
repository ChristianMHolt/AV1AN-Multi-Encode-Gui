import sys
import signal
from ui.qt import QtWidgets
from ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AV1 Encoder Pro")
    app.setOrganizationName("AV1Runner")
    
    win = MainWindow()
    win.show()
    
    # Allow Ctrl+C to kill
    signal.signal(signal.SIGINT, lambda *args: win.close())
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
