import os
try:
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                                   QTabWidget, QWidget, QGroupBox, QLabel, QLineEdit, 
                                   QCheckBox, QSpinBox, QComboBox, QFileDialog)
    from PySide6.QtCore import QSettings
except ImportError:
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                                 QTabWidget, QWidget, QGroupBox, QLabel, QLineEdit, 
                                 QCheckBox, QSpinBox, QComboBox, QFileDialog)
    from PyQt6.QtCore import QSettings

from config import (
    DEFAULT_OUT_DIR, DEFAULT_SVT_PATH, DEFAULT_TEMP_DIR, USE_CHUNK_METHOD
)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(600, 450)
        self.settings = QSettings("AV1Runner", "EncoderPro")
        
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        tabs.addTab(self._gen_tab(), "General")
        tabs.addTab(self._enc_tab(), "Encoder")
        tabs.addTab(self._adv_tab(), "Advanced")
        layout.addWidget(tabs)
        
        btns = QHBoxLayout()
        bsave = QPushButton("Save")
        bsave.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(bsave)
        btns.addWidget(QPushButton("Cancel", clicked=self.reject))
        layout.addLayout(btns)
        
        self.load()
        
    def _gen_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        
        g = QGroupBox("Output")
        gl = QVBoxLayout(g)
        h = QHBoxLayout()
        h.addWidget(QLabel("Dir:"))
        self.out_edit = QLineEdit()
        h.addWidget(self.out_edit)
        btn = QPushButton("..."); btn.clicked.connect(lambda: self._browse(self.out_edit))
        h.addWidget(btn)
        gl.addLayout(h)
        self.chk_clean = QCheckBox("Auto cleanup temp")
        gl.addWidget(self.chk_clean)
        l.addWidget(g)
        
        g2 = QGroupBox("Notifications")
        gl2 = QVBoxLayout(g2)
        self.chk_done = QCheckBox("Notify on Complete")
        self.chk_err = QCheckBox("Notify on Error")
        self.chk_snd = QCheckBox("Play Sound")
        self.chk_graph = QCheckBox("Disable Graphs")
        gl2.addWidget(self.chk_done); gl2.addWidget(self.chk_err)
        gl2.addWidget(self.chk_snd); gl2.addWidget(self.chk_graph)
        l.addWidget(g2)
        l.addStretch()
        return w

    def _enc_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        
        h = QHBoxLayout()
        h.addWidget(QLabel("SVT Path:"))
        self.svt_edit = QLineEdit()
        h.addWidget(self.svt_edit)
        btn = QPushButton("..."); btn.clicked.connect(lambda: self._browse(self.svt_edit, file=True))
        h.addWidget(btn)
        l.addLayout(h)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Temp Dir:"))
        self.tmp_edit = QLineEdit()
        h2.addWidget(self.tmp_edit)
        btn2 = QPushButton("..."); btn2.clicked.connect(lambda: self._browse(self.tmp_edit))
        h2.addWidget(btn2)
        l.addLayout(h2)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Chunk Method:"))
        self.chunk_combo = QComboBox()
        # Changed to valid options, defaulting to select
        self.chunk_combo.addItems(["select", "hybrid", "vs_ffms2", "vs_lsmash"])
        h3.addWidget(self.chunk_combo)
        l.addLayout(h3)
        l.addStretch()
        return w

    def _adv_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        self.chk_resume = QCheckBox("Resume")
        self.chk_keep = QCheckBox("Keep intermediate")
        
        h = QHBoxLayout()
        h.addWidget(QLabel("Retries:"))
        self.sp_retry = QSpinBox(); self.sp_retry.setRange(0, 5)
        h.addWidget(self.sp_retry)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Disk Warn (GB):"))
        self.sp_disk = QSpinBox(); self.sp_disk.setRange(1, 1000)
        h2.addWidget(self.sp_disk)
        
        l.addWidget(self.chk_resume); l.addWidget(self.chk_keep)
        l.addLayout(h); l.addLayout(h2)
        l.addStretch()
        return w

    def _browse(self, field, file=False):
        if file:
            p, _ = QFileDialog.getOpenFileName(self, "Select File")
        else:
            p = QFileDialog.getExistingDirectory(self, "Select Dir")
        if p: field.setText(p)

    def load(self):
        s = self.settings
        self.out_edit.setText(s.value("output_dir", str(DEFAULT_OUT_DIR)))
        self.svt_edit.setText(s.value("svt_path", DEFAULT_SVT_PATH))
        self.tmp_edit.setText(s.value("temp_dir", DEFAULT_TEMP_DIR))
        self.chunk_combo.setCurrentText(s.value("chunk_method", USE_CHUNK_METHOD))
        self.chk_clean.setChecked(s.value("auto_cleanup", True, type=bool))
        self.chk_done.setChecked(s.value("notif_complete", True, type=bool))
        self.chk_err.setChecked(s.value("notif_error", True, type=bool))
        self.chk_snd.setChecked(s.value("play_sound", False, type=bool))
        self.chk_graph.setChecked(s.value("disable_graphs", False, type=bool))
        self.chk_resume.setChecked(s.value("resume", True, type=bool))
        self.chk_keep.setChecked(s.value("keep", True, type=bool))
        self.sp_retry.setValue(int(s.value("max_retries", 2)))
        self.sp_disk.setValue(int(s.value("disk_warn_gb", 50)))

    def save_settings(self):
        s = self.settings
        s.setValue("output_dir", self.out_edit.text())
        s.setValue("svt_path", self.svt_edit.text())
        s.setValue("temp_dir", self.tmp_edit.text())
        s.setValue("chunk_method", self.chunk_combo.currentText())
        s.setValue("auto_cleanup", self.chk_clean.isChecked())
        s.setValue("notif_complete", self.chk_done.isChecked())
        s.setValue("notif_error", self.chk_err.isChecked())
        s.setValue("play_sound", self.chk_snd.isChecked())
        s.setValue("disable_graphs", self.chk_graph.isChecked())
        s.setValue("resume", self.chk_resume.isChecked())
        s.setValue("keep", self.chk_keep.isChecked())
        s.setValue("max_retries", self.sp_retry.value())
        s.setValue("disk_warn_gb", self.sp_disk.value())

    def get_config(self):
        return {
            "output_dir": self.out_edit.text(),
            "svt_path": self.svt_edit.text(),
            "temp_dir": self.tmp_edit.text(),
            "chunk_method": self.chunk_combo.currentText(),
            "auto_cleanup": self.chk_clean.isChecked(),
            "notif_complete": self.chk_done.isChecked(),
            "notif_error": self.chk_err.isChecked(),
            "play_sound": self.chk_snd.isChecked(),
            "disable_graphs": self.chk_graph.isChecked(),
            "resume": self.chk_resume.isChecked(),
            "keep": self.chk_keep.isChecked(),
            "max_retries": self.sp_retry.value(),
            "disk_warn_gb": self.sp_disk.value(),
        }