import os
from .qt import QtWidgets, QtCore

from config import (
    DEFAULT_OUT_DIR, DEFAULT_IN_DIR, DEFAULT_SVT_PATH, DEFAULT_TEMP_DIR, USE_CHUNK_METHOD
)

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(600, 450)
        self.settings = QtCore.QSettings("AV1Runner", "EncoderPro")
        
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        
        tabs.addTab(self._gen_tab(), "General")
        tabs.addTab(self._enc_tab(), "Encoder")
        tabs.addTab(self._adv_tab(), "Advanced")
        layout.addWidget(tabs)
        
        btns = QtWidgets.QHBoxLayout()
        bsave = QtWidgets.QPushButton("Save")
        bsave.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(bsave)
        btns.addWidget(QtWidgets.QPushButton("Cancel", clicked=self.reject))
        layout.addLayout(btns)
        
        self.load()
        
    def _gen_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        
        g = QtWidgets.QGroupBox("Directories")
        gl = QtWidgets.QVBoxLayout(g)

        # Input
        h_in = QtWidgets.QHBoxLayout()
        h_in.addWidget(QtWidgets.QLabel("Input Dir:"))
        self.in_edit = QtWidgets.QLineEdit()
        h_in.addWidget(self.in_edit)
        btn_in = QtWidgets.QPushButton("..."); btn_in.clicked.connect(lambda: self._browse(self.in_edit))
        h_in.addWidget(btn_in)
        gl.addLayout(h_in)

        # Output
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Output Dir:"))
        self.out_edit = QtWidgets.QLineEdit()
        h.addWidget(self.out_edit)
        btn = QtWidgets.QPushButton("..."); btn.clicked.connect(lambda: self._browse(self.out_edit))
        h.addWidget(btn)
        gl.addLayout(h)

        self.chk_clean = QtWidgets.QCheckBox("Auto cleanup temp")
        gl.addWidget(self.chk_clean)
        l.addWidget(g)
        
        g2 = QtWidgets.QGroupBox("Notifications")
        gl2 = QtWidgets.QVBoxLayout(g2)
        self.chk_done = QtWidgets.QCheckBox("Notify on Complete")
        self.chk_err = QtWidgets.QCheckBox("Notify on Error")
        self.chk_snd = QtWidgets.QCheckBox("Play Sound")
        self.chk_graph = QtWidgets.QCheckBox("Disable Graphs")
        gl2.addWidget(self.chk_done); gl2.addWidget(self.chk_err)
        gl2.addWidget(self.chk_snd); gl2.addWidget(self.chk_graph)
        l.addWidget(g2)
        l.addStretch()
        return w

    def _enc_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("SVT Path:"))
        self.svt_edit = QtWidgets.QLineEdit()
        h.addWidget(self.svt_edit)
        btn = QtWidgets.QPushButton("..."); btn.clicked.connect(lambda: self._browse(self.svt_edit, file=True))
        h.addWidget(btn)
        l.addLayout(h)
        
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Temp Dir:"))
        self.tmp_edit = QtWidgets.QLineEdit()
        h2.addWidget(self.tmp_edit)
        btn2 = QtWidgets.QPushButton("..."); btn2.clicked.connect(lambda: self._browse(self.tmp_edit))
        h2.addWidget(btn2)
        l.addLayout(h2)
        
        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(QtWidgets.QLabel("Chunk Method:"))
        self.chunk_combo = QtWidgets.QComboBox()
        self.chunk_combo.addItems(["select", "hybrid", "vs_ffms2", "vs_lsmash"])
        h3.addWidget(self.chunk_combo)
        l.addLayout(h3)
        l.addStretch()
        return w

    def _adv_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        self.chk_resume = QtWidgets.QCheckBox("Resume")
        self.chk_keep = QtWidgets.QCheckBox("Keep intermediate")
        
        # --- NEW: VMAF Toggle ---
        self.chk_vmaf = QtWidgets.QCheckBox("Calculate VMAF Score (Slows down completion)")
        self.chk_vmaf.setToolTip("Runs a quality check after encoding. Useful for benchmarking.")
        
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Retries:"))
        self.sp_retry = QtWidgets.QSpinBox(); self.sp_retry.setRange(0, 5)
        h.addWidget(self.sp_retry)
        
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Disk Warn (GB):"))
        self.sp_disk = QtWidgets.QSpinBox(); self.sp_disk.setRange(1, 1000)
        h2.addWidget(self.sp_disk)
        
        l.addWidget(self.chk_resume); l.addWidget(self.chk_keep)
        l.addWidget(self.chk_vmaf) # Added here
        l.addLayout(h); l.addLayout(h2)
        l.addStretch()
        return w

    def _browse(self, field, file=False):
        if file:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File")
        else:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dir")
        if p: field.setText(p)

    def load(self):
        s = self.settings
        self.in_edit.setText(s.value("input_dir", str(DEFAULT_IN_DIR)))
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
        
        # Load VMAF setting (Default True)
        self.chk_vmaf.setChecked(s.value("calc_vmaf", True, type=bool))
        
        self.sp_retry.setValue(int(s.value("max_retries", 2)))
        self.sp_disk.setValue(int(s.value("disk_warn_gb", 50)))

    def save_settings(self):
        s = self.settings
        s.setValue("input_dir", self.in_edit.text())
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
        s.setValue("calc_vmaf", self.chk_vmaf.isChecked())
        s.setValue("max_retries", self.sp_retry.value())
        s.setValue("disk_warn_gb", self.sp_disk.value())

    def get_config(self):
        return {
            "input_dir": self.in_edit.text(),
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
            "calc_vmaf": self.chk_vmaf.isChecked(), # <--- Exported
            "max_retries": self.sp_retry.value(),
            "disk_warn_gb": self.sp_disk.value(),
        }
