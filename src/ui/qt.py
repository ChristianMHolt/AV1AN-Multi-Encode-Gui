from .qt_core import Signal, Slot, QObject, QThread, QTimer, QSettings, QtCore

if QtCore.__name__.startswith("PySide6"):
    from PySide6 import QtGui, QtWidgets
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    from PySide6.QtWidgets import *
else:
    from PyQt6 import QtGui, QtWidgets
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *
