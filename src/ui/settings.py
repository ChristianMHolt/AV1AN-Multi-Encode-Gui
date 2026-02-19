import json
from .qt import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                 QTabWidget, QWidget, QGroupBox, QLabel, QLineEdit,
                 QCheckBox, QSpinBox, QComboBox, QFileDialog, QListWidget,
                 QListWidgetItem, QMessageBox, QSettings)

from config import (
    DEFAULT_OUT_DIR, DEFAULT_IN_DIR, DEFAULT_SVT_PATH, DEFAULT_TEMP_DIR, USE_CHUNK_METHOD
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
        tabs.addTab(self._presets_tab(), "Presets")
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
        
        g = QGroupBox("Directories")
        gl = QVBoxLayout(g)

        # Input
        h_in = QHBoxLayout()
        h_in.addWidget(QLabel("Input Dir:"))
        self.in_edit = QLineEdit()
        h_in.addWidget(self.in_edit)
        btn_in = QPushButton("..."); btn_in.clicked.connect(lambda: self._browse(self.in_edit))
        h_in.addWidget(btn_in)
        gl.addLayout(h_in)

        # Output
        h = QHBoxLayout()
        h.addWidget(QLabel("Output Dir:"))
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
        self.chunk_combo.addItems(["select", "hybrid", "vs_ffms2", "vs_lsmash"])
        h3.addWidget(self.chunk_combo)
        l.addLayout(h3)
        l.addStretch()
        return w

    def _adv_tab(self):
        w = QWidget(); l = QVBoxLayout(w)
        self.chk_resume = QCheckBox("Resume")
        self.chk_keep = QCheckBox("Keep intermediate")
        
        # --- NEW: VMAF Toggle ---
        self.chk_vmaf = QCheckBox("Calculate VMAF Score (Slows down completion)")
        self.chk_vmaf.setToolTip("Runs a quality check after encoding. Useful for benchmarking.")
        
        h = QHBoxLayout()
        h.addWidget(QLabel("Retries:"))
        self.sp_retry = QSpinBox(); self.sp_retry.setRange(0, 5)
        h.addWidget(self.sp_retry)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Disk Warn (GB):"))
        self.sp_disk = QSpinBox(); self.sp_disk.setRange(1, 1000)
        h2.addWidget(self.sp_disk)
        
        l.addWidget(self.chk_resume); l.addWidget(self.chk_keep)
        l.addWidget(self.chk_vmaf) # Added here
        l.addLayout(h); l.addLayout(h2)
        l.addStretch()
        return w

    def _presets_tab(self):
        w = QWidget(); l = QVBoxLayout(w)

        self.preset_list = QListWidget()
        l.addWidget(QLabel("Custom Presets:"))
        l.addWidget(self.preset_list)

        g = QGroupBox("Edit Preset")
        gl = QVBoxLayout(g)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Name:"))
        self.p_name = QLineEdit()
        h1.addWidget(self.p_name)
        gl.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("SVT Opts:"))
        self.p_opts = QLineEdit()
        h2.addWidget(self.p_opts)
        gl.addLayout(h2)

        btns = QHBoxLayout()
        b_add = QPushButton("Add/Update")
        b_add.clicked.connect(self.add_preset)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self.del_preset)
        btns.addWidget(b_add); btns.addWidget(b_del)
        gl.addLayout(btns)

        l.addWidget(g)
        return w

    def add_preset(self):
        name = self.p_name.text().strip()
        opts = self.p_opts.text().strip()
        if not name or not opts: return

        self.custom_presets[name] = {"svt_opts": opts, "workers": "auto"}
        self._refresh_preset_list()
        self.p_name.clear(); self.p_opts.clear()

    def del_preset(self):
        item = self.preset_list.currentItem()
        if item:
            # Assuming format "Name (opts)"
            # Better to store key in item data
            name = item.data(100) # Use user role
            if name in self.custom_presets:
                del self.custom_presets[name]
                self._refresh_preset_list()

    def _refresh_preset_list(self):
        self.preset_list.clear()
        for k, v in self.custom_presets.items():
            item = QListWidgetItem(f"{k} ({v['svt_opts']})")
            item.setData(100, k)
            self.preset_list.addItem(item)

    def _browse(self, field, file=False):
        if file:
            p, _ = QFileDialog.getOpenFileName(self, "Select File")
        else:
            p = QFileDialog.getExistingDirectory(self, "Select Dir")
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

        # Load Custom Presets
        try:
            raw = self.settings.value("custom_presets", "{}")
            self.custom_presets = json.loads(raw)
        except:
            self.custom_presets = {}
        # We need to call refresh but QListWidgetItem needs imports which might be tricky if handled inline.
        # I'll handle imports in _refresh_preset_list properly.
        self._refresh_preset_list()

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
        s.setValue("custom_presets", json.dumps(self.custom_presets))

    def get_custom_presets(self):
        return dict(self.custom_presets)

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