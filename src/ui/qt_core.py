try:
    from PySide6 import QtCore
    from PySide6.QtCore import Signal, Slot, QObject, QThread, QTimer, QSettings
except ImportError:
    from PyQt6 import QtCore
    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, QObject, QThread, QTimer, QSettings
