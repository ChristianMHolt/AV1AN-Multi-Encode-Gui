import time
from typing import Callable
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QProgressBar, QPushButton, QDialog, QComboBox, 
                                   QLineEdit, QTextEdit, QTabWidget)
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
                                 QProgressBar, QPushButton, QDialog, QComboBox, 
                                 QLineEdit, QTextEdit, QTabWidget)

from config import DEFAULT_PRESETS
from models import Job, JobStatus

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
    pg.setConfigOptions(antialias=True, useOpenGL=False)
except:
    HAS_PYQTGRAPH = False

class JobTile(QFrame):
    def __init__(self, job: Job, on_toggle: Callable, on_remove: Callable, 
                 on_log: Callable, disable_graphs: bool = False, parent=None):
        super().__init__(parent)
        self.job = job
        self.on_toggle = on_toggle
        self.on_remove = on_remove
        self.on_log = on_log
        self.disable_graphs = disable_graphs
        
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #3a3a3a; border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:1 #252525);
                padding: 10px;
            }
            QFrame:hover { border: 1px solid #4a4a4a; }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(2)
        
        h = QHBoxLayout()
        self.title = QLabel(job.infile.name)
        self.title.setStyleSheet("font-weight: bold; font-size: 10pt;")
        h.addWidget(self.title, 1)
        self.status = QLabel(job.status.value)
        self.status.setStyleSheet("color: #888;")
        h.addWidget(self.status)
        self.layout.addLayout(h)
        
        self.info = QLabel(f"Managed Auto-Scaling • {job.preset_name}")
        self.info.setStyleSheet("color: #aaa; font-size: 8pt;")
        self.layout.addWidget(self.info)
        
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555; border-radius: 4px; text-align: center;
                background: #1a1a1a; height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a90e2, stop:1 #357abd);
                border-radius: 3px;
            }
        """)
        self.layout.addWidget(self.bar)
        
        self.stats = QLabel("—")
        self.stats.setStyleSheet("color: #bbb; font-size: 8pt;")
        self.layout.addWidget(self.stats)
        
        if HAS_PYQTGRAPH and not disable_graphs:
            self.plot = pg.PlotWidget()
            self.plot.setBackground('#1a1a1a')
            self.plot.setFixedHeight(60)
            self.plot.hideAxis('bottom')
            self.plot.getAxis('left').setPen('#888')
            self.curve = self.plot.plot([], [], pen=pg.mkPen('#4a90e2', width=2))
            self.layout.addWidget(self.plot)
        else:
            self.plot = None
            
        btns = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedWidth(80)
        self.btn_pause.clicked.connect(lambda: self.on_toggle(self.job.idx))
        
        self.btn_log = QPushButton("Log")
        self.btn_log.setFixedWidth(80)
        self.btn_log.clicked.connect(lambda: self.on_log(self.job.idx))
        
        self.btn_rm = QPushButton("Remove")
        self.btn_rm.setFixedWidth(80)
        self.btn_rm.clicked.connect(lambda: self.on_remove(self.job.idx))
        
        btns.addWidget(self.btn_pause)
        btns.addWidget(self.btn_log)
        btns.addWidget(self.btn_rm)
        btns.addStretch()
        self.layout.addLayout(btns)
        
        self.update_ui()

    def update_ui(self):
        j = self.job
        self.bar.setValue(int(j.pct * 10))
        self.bar.setFormat(f"{j.pct:.1f}%")

        if j.status_text:
            self.status.setText(j.status_text)
        else:
            self.status.setText(j.status.value)
        
        can_pause = (j.status == JobStatus.RUNNING and j.proc is not None)
        can_resume = (j.status == JobStatus.PAUSED)
        
        self.btn_pause.setEnabled(bool(can_pause or can_resume))
        self.btn_pause.setText("Resume" if j.status == JobStatus.PAUSED else "Pause")
        self.btn_rm.setEnabled(j.status in [JobStatus.QUEUED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
        
        parts = []
        if j.status == JobStatus.RUNNING:
            parts.append(f"FPS: {j.current_fps:.1f} (avg: {j.avg_fps:.1f})")
            if j.eta_seconds:
                if j.eta_seconds < 60:
                    parts.append(f"ETA: {int(j.eta_seconds)}s")
                else:
                    mins = j.eta_seconds / 60.0
                    parts.append(f"ETA: {mins:.1f}m")
                    
        elif j.status == JobStatus.MUXING:
            parts.append("Muxing final file...")
            
        elif j.status == JobStatus.VMAF:
            parts.append(f"VMAF Analysis: {j.current_fps:.1f} fps")
            if j.eta_seconds:
                if j.eta_seconds < 60:
                    parts.append(f"ETA: {int(j.eta_seconds)}s")
                else:
                    mins = j.eta_seconds / 60.0
                    parts.append(f"ETA: {mins:.1f}m")
                    
        elif j.status == JobStatus.COMPLETED:
            parts.append("Done")
            if j.encoded_size > 0:
                ratio = (1 - j.encoded_size / j.original_size) * 100
                parts.append(f"Saved {ratio:.1f}%")
            if j.vmaf_score > 0:
                lows = ""
                if j.vmaf_1_percent > 0:
                    lows = f" (1%: {j.vmaf_1_percent:.1f} | 0.1%: {j.vmaf_01_percent:.1f})"
                parts.append(f"VMAF: {j.vmaf_score:.2f}{lows}")
        
        self.stats.setText(" • ".join(parts))
        
        if self.plot and j.fps_hist:
            self.curve.setData(list(j.fps_hist))

class LogViewer(QDialog):
    def __init__(self, job: Job, parent=None):
        super().__init__(parent)
        self.job = job
        self.setWindowTitle(f"Logs - {job.infile.name}")
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        self.txt_enc = QTextEdit(); self.txt_enc.setReadOnly(True)
        self.txt_mux = QTextEdit(); self.txt_mux.setReadOnly(True)
        self.txt_vmaf = QTextEdit(); self.txt_vmaf.setReadOnly(True) # <--- NEW TAB
        
        self.txt_enc.setFont(QtGui.QFont("Courier", 9))
        self.txt_mux.setFont(QtGui.QFont("Courier", 9))
        self.txt_vmaf.setFont(QtGui.QFont("Courier", 9))
        
        tabs.addTab(self.txt_enc, "Encoding Log")
        tabs.addTab(self.txt_mux, "Muxing Log")
        tabs.addTab(self.txt_vmaf, "VMAF Log")
        layout.addWidget(tabs)
        
        btn = QPushButton("Refresh")
        btn.clicked.connect(self.load)
        layout.addWidget(btn)
        self.load()
        
    def load(self):
        try:
            if self.job.term_log.exists():
                with open(self.job.term_log, "r", encoding="utf-8", errors="replace") as f:
                    self.txt_enc.setPlainText(f.read())
            if self.job.mux_log.exists():
                with open(self.job.mux_log, "r", encoding="utf-8", errors="replace") as f:
                    self.txt_mux.setPlainText(f.read())
            # --- Load VMAF Log ---
            if self.job.vmaf_log.exists():
                with open(self.job.vmaf_log, "r", encoding="utf-8", errors="replace") as f:
                    self.txt_vmaf.setPlainText(f.read())
            else:
                self.txt_vmaf.setPlainText(f"File not found: {self.job.vmaf_log}")
        except: pass