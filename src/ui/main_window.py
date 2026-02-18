import sys
import signal
import csv
from pathlib import Path
from .qt import QtCore, QtGui, QtWidgets, Slot

from config import DEFAULT_PRESETS, INPUT_GLOBS, IS_WINDOWS, DEFAULT_OUT_DIR, DEFAULT_IN_DIR
from worker import Runner, SystemMonitor, get_missing_tools, format_size
from models import JobStatus
from .widgets import JobTile, LogViewer
from .settings import SettingsDialog

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AV1 Encoder Pro - Auto Scale")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        self.setup_theme()
        
        # Init Settings
        self.settings_dlg = SettingsDialog(self)
        self.config = self.settings_dlg.get_config()
        
        # Check Tools
        miss = get_missing_tools()
        if miss:
            QtWidgets.QMessageBox.warning(self, "Missing Tools", "\n".join(f"{t}: {h}" for t,h in miss))
        
        # Init Core
        self.runner = Runner(self.config, self)
        self.sys_mon = SystemMonitor()
        
        # UI Setup
        self.setup_ui()
        self.connect_signals()
        
        # Start
        self.sys_mon.start()
        self.load_initial()
        
    def setup_theme(self):
        QtWidgets.QApplication.setStyle("Fusion")
        p = QtGui.QPalette()
        c = QtGui.QColor
        p.setColor(QtGui.QPalette.ColorRole.Window, c(35,35,35))
        p.setColor(QtGui.QPalette.ColorRole.WindowText, c(255,255,255))
        p.setColor(QtGui.QPalette.ColorRole.Base, c(25,25,25))
        p.setColor(QtGui.QPalette.ColorRole.AlternateBase, c(35,35,35))
        p.setColor(QtGui.QPalette.ColorRole.Text, c(255,255,255))
        p.setColor(QtGui.QPalette.ColorRole.Button, c(45,45,45))
        p.setColor(QtGui.QPalette.ColorRole.ButtonText, c(255,255,255))
        p.setColor(QtGui.QPalette.ColorRole.Highlight, c(42,130,218))
        QtWidgets.QApplication.setPalette(p)

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Top Bar
        h = QtWidgets.QHBoxLayout()
        self.lbl_fps = QtWidgets.QLabel("Total FPS: 0.0")
        self.lbl_fps.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 14px;")
        h.addWidget(self.lbl_fps)
        h.addStretch()
        
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(list(DEFAULT_PRESETS.keys()))
        h.addWidget(QtWidgets.QLabel("Preset:"))
        h.addWidget(self.combo_preset)
        
        b_add = QtWidgets.QPushButton("Add Files"); b_add.clicked.connect(self.add_files_dlg)
        b_set = QtWidgets.QPushButton("Settings"); b_set.clicked.connect(self.show_settings)
        b_exp = QtWidgets.QPushButton("Export"); b_exp.clicked.connect(self.export_csv)
        h.addWidget(b_add); h.addWidget(b_set); h.addWidget(b_exp)
        layout.addLayout(h)
        
        # Splitter
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        
        # Jobs Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.job_cont = QtWidgets.QWidget()
        self.job_layout = QtWidgets.QVBoxLayout(self.job_cont)
        self.job_layout.addStretch()
        scroll.setWidget(self.job_cont)
        split.addWidget(scroll)
        
        # Stats Panel
        stats = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(stats)
        
        g_sys = QtWidgets.QGroupBox("System")
        l_sys = QtWidgets.QVBoxLayout(g_sys)
        self.l_cpu = QtWidgets.QLabel("CPU: -")
        self.l_mem = QtWidgets.QLabel("RAM: -")
        l_sys.addWidget(self.l_cpu); l_sys.addWidget(self.l_mem)
        sl.addWidget(g_sys)
        
        g_job = QtWidgets.QGroupBox("Jobs")
        l_job = QtWidgets.QVBoxLayout(g_job)
        self.l_q = QtWidgets.QLabel("Queued: 0")
        self.l_r = QtWidgets.QLabel("Running: 0")
        self.l_d = QtWidgets.QLabel("Done: 0")
        l_job.addWidget(self.l_q); l_job.addWidget(self.l_r); l_job.addWidget(self.l_d)
        sl.addWidget(g_job)
        
        split.addWidget(stats)
        
        # --- FIX: Set initial sizes [Top, Bottom] ---
        # 10000 vs 1 ensures the top gets all available space 
        # while the bottom shrinks to its minimum requirements.
        split.setSizes([10000, 1]) 
        
        layout.addWidget(split)
        
        self.tiles = {}
        
    def connect_signals(self):
        self.runner.job_updated.connect(self.on_job_upd)
        self.runner.job_finished.connect(self.on_job_fin)
        self.runner.total_fps_changed.connect(lambda f: self.lbl_fps.setText(f"FPS: {f:.1f}"))
        self.runner.shutdown_complete.connect(self.final_exit)
        self.sys_mon.stats_updated.connect(self.on_sys_upd)

    def load_initial(self):
        files = []
        input_dir = Path(self.config.get("input_dir", DEFAULT_IN_DIR))
        if not input_dir.exists():
            input_dir.mkdir(parents=True, exist_ok=True)
        for g in INPUT_GLOBS: files.extend(input_dir.glob(g))
        if files: self.add_to_runner(files)

    def add_files_dlg(self):
        input_dir = self.config.get("input_dir", str(DEFAULT_IN_DIR))
        fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add Files", input_dir, "Video (*.mkv *.mp4 *.avi *.ts *.webm);;All (*)")
        if fs: self.add_to_runner([Path(f) for f in fs])

    def add_to_runner(self, paths):
        jobs = self.runner.add_files(paths, self.combo_preset.currentText())
        for j in jobs:
            t = JobTile(j, self.runner.toggle_pause, self.rm_job, self.show_log, 
                        self.config.get("disable_graphs", False))
            self.tiles[j.idx] = t
            self.job_layout.insertWidget(self.job_layout.count()-1, t)
        self.upd_stats()

    def rm_job(self, idx):
        if self.runner.remove_job(idx):
            t = self.tiles.pop(idx, None)
            if t: t.deleteLater()
            self.upd_stats()
        else:
            QtWidgets.QMessageBox.warning(self, "Busy", "Stop job first.")

    def show_log(self, idx):
        LogViewer(self.runner.jobs[idx], self).exec()

    def show_settings(self):
        if self.settings_dlg.exec():
            self.settings_dlg.save_settings()
            self.config = self.settings_dlg.get_config()
            self.runner.update_config(self.config)

    def export_csv(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", "stats.csv", "CSV (*.csv)")
        if p:
            try:
                with open(p, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(["File", "Status", "Preset", "FPS"])
                    for j in self.runner.jobs:
                        w.writerow([j.infile.name, j.status.value, j.preset_name, f"{j.avg_fps:.2f}"])
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

    @Slot(int)
    def on_job_upd(self, idx):
        if idx in self.tiles: self.tiles[idx].update_ui()
        self.upd_stats()

    @Slot(int)
    def on_job_fin(self, idx):
        self.on_job_upd(idx)
        j = self.runner.jobs[idx]
        if j.status == JobStatus.FAILED and self.config["notif_error"]:
            self.statusBar().showMessage(f"Failed: {j.infile.name}")
        elif j.status == JobStatus.COMPLETED and self.config["notif_complete"]:
            self.statusBar().showMessage(f"Done: {j.infile.name}")

    @Slot(dict)
    def on_sys_upd(self, s):
        if "cpu_percent" in s: self.l_cpu.setText(f"CPU: {s['cpu_percent']:.1f}%")
        if "mem_percent" in s: self.l_mem.setText(f"RAM: {s['mem_percent']:.1f}%")

    def upd_stats(self):
        c = {s:0 for s in JobStatus}
        for j in self.runner.jobs: c[j.status] += 1
        self.l_q.setText(f"Queued: {c[JobStatus.QUEUED]}")
        self.l_r.setText(f"Running: {c[JobStatus.RUNNING]}")
        self.l_d.setText(f"Done: {c[JobStatus.COMPLETED]}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()
        
    def dropEvent(self, e):
        fs = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        valid = [f for f in fs if f.suffix.lower() in [".mkv",".mp4",".avi",".ts",".webm"]]
        if valid: self.add_to_runner(valid)

    def closeEvent(self, e):
        if self.runner._closing:
            e.accept(); return
        
        active = any(j.status in [JobStatus.RUNNING, JobStatus.MUXING] for j in self.runner.jobs)
        if active:
            if QtWidgets.QMessageBox.question(self, "Exit", "Stop encoding?") != QtWidgets.QMessageBox.StandardButton.Yes:
                e.ignore(); return
        
        e.ignore()
        self.setEnabled(False)
        self.statusBar().showMessage("Stopping...")
        self.runner.request_stop_all()

    @Slot()
    def final_exit(self):
        self.sys_mon.stop()
        QtWidgets.QApplication.quit()
