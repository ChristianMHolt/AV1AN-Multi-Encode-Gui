import sys

# Common Qt imports
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Slot, Signal, QThread, QObject, QTimer, QSettings
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import pyqtSlot as Slot, pyqtSignal as Signal, QThread, QObject, QTimer, QSettings

# Export these for other modules to use
__all__ = [
    'QtCore', 'QtGui', 'QtWidgets',
    'Slot', 'Signal', 'QThread', 'QObject', 'QTimer', 'QSettings'
]
