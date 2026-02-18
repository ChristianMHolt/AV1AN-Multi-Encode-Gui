try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Slot, Signal
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import pyqtSlot as Slot, pyqtSignal as Signal
