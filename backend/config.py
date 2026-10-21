"""Central configuration. Every setting can be overridden with an environment variable."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]            # where the code (and index.html) lives


def _load_dotenv():
    """Tiny .env reader (no extra package): lines like ANTHROPIC_API_KEY=sk-ant-..."""
    f = APP_DIR / ".env"
    if os.getenv("DEMANDAI_DISABLE_DOTENV") or not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
ROOT = Path(os.getenv("DEMANDAI_ROOT", APP_DIR))         # where data/models/outputs live
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS, PREPROCESSING = ROOT / "models", ROOT / "models" / "preprocessing"
FORECASTS, METRICS, REPORTS = (ROOT / "outputs" / n for n in ("forecasts", "metrics", "reports"))

HORIZON = 28                                             # forecast horizon in days
STORES = os.getenv("DEMANDAI_STORES", "CA_1,TX_1,WI_1").split(",")
ITEMS_PER_CAT = int(os.getenv("DEMANDAI_ITEMS_PER_CAT", 30))
HISTORY_DAYS = int(os.getenv("DEMANDAI_HISTORY_DAYS", 730))
SEED = 42

LSTM_WINDOW = 56
LSTM_EPOCHS = int(os.getenv("DEMANDAI_LSTM_EPOCHS", 30))
LSTM_STRIDE = int(os.getenv("DEMANDAI_LSTM_STRIDE", 2))

LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY")             # optional, server-side only
LLM_MODEL = os.getenv("DEMANDAI_LLM_MODEL", "claude-sonnet-5-5")

LONG_FILE = PROCESSED / "sales_long.parquet"
DB_FILE = PROCESSED / "demandai.db"
MODEL_ORDER = ["seasonal_naive", "xgboost", "lstm"]


def ensure_dirs():
    for p in (RAW, PROCESSED, PREPROCESSING, FORECASTS, METRICS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
