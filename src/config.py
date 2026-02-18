import os
import re
from pathlib import Path
import platform

IS_WINDOWS = (os.name == "nt") or (platform.system().lower() == "windows")

# Calculate the project root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# --- POINT TO THE RUST EXE ---
DEFAULT_AV1AN_PATH = str(SRC_DIR / "av1an.exe")
DEFAULT_SVT_PATH = str(SRC_DIR / "SvtAv1EncApp.exe")

DEFAULT_TEMP_DIR = str(PROJECT_ROOT / "temp")

DEFAULT_OUT_DIR = PROJECT_ROOT / "output"
DEFAULT_IN_DIR = PROJECT_ROOT / "input"

LOG_DIR = (PROJECT_ROOT / ".log").resolve()

# Engine Config
MAX_JOBS_CAP = 4 
BLOCKED_CPUS = [0, 16] 

# FAST METHOD (Rust handles this natively)
USE_CHUNK_METHOD = "ffms2"

INPUT_DIR = DEFAULT_IN_DIR
INPUT_GLOBS = ["*.mkv", "*.mp4", "*.mov", "*.avi", "*.m2ts", "*.ts", "*.webm"]
FPS_WINDOW = 60
GUI_REFRESH_HZ = 8
STOP_GRACE_SEC = 5.0
TERM_GRACE_SEC = 3.0

DEFAULT_PRESETS = {
    "Ultra Quality": {
        "svt_opts": "--preset 3 --rc 0 --crf 12 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.2 --sharp-tx 1 --noise-adaptive-filtering 1 --enable-qm 1 --qm-min 8 --qp-scale-compress-strength 3 --noise-norm-strength 1 --fast-decode 1 --enable-dlf 2",
        "passes": 1,
        "workers": "auto",
    },
    "High Quality": {
        "svt_opts": "--preset 3 --rc 0 --crf 12 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.2 --sharp-tx 1 --noise-adaptive-filtering 1 --enable-qm 1 --qm-min 8 --qp-scale-compress-strength 3 --noise-norm-strength 1 --fast-decode 1 --enable-dlf 2",
        "passes": 1,
        "workers": "auto",
    },
    "Balanced": {
        "svt_opts": "--preset 4 --rc 0 --crf 16 --aq-mode 2 --keyint 48 --enable-tf 0 --tune 0 --psy-rd 1.0 --enable-qm 1 --qm-min 0 --fast-decode 1",
        "passes": 1,
        "workers": "auto",
    },
    "Fast": {
        "svt_opts": "--preset 6 --rc 0 --crf 20 --aq-mode 1 --keyint 60 --fast-decode 1",
        "passes": 1,
        "workers": "auto",
    },
}

# Regex Patterns
IGNORE_FPS_LINE = re.compile(r"\b(Video:|Stream #|Input #)\b", re.IGNORECASE)
FR_S_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps", re.IGNORECASE)
SPEED_FPS_RE = re.compile(r"\b(speed|enc|encoding)[^\n]*?(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
S_PER_FR_RE  = re.compile(r"([0-9]+(?:\.\d+)?)\s*s/fr", re.IGNORECASE)
PCT_RE = re.compile(r"(\d+)\s*/\s*(\d+)", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# AV1AN Progress
AV1AN_PROGRESS_RE = re.compile(r"PROGRESS:\s+(\d+(?:\.\d+)?)%\s+\|\s+Frames:\s+(\d+)/(\d+)\s+\|\s+Chunks:\s+(\d+)/(\d+)\s+\|\s+Speed:\s+(\d+(?:\.\d+)?)\s+fps", re.IGNORECASE)

# FFmpeg Progress (for VMAF)
# Matches: frame=  123 fps= 24.5 ...
FFMPEG_PROGRESS_RE = re.compile(r"frame=\s*(\d+).*?fps=\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
