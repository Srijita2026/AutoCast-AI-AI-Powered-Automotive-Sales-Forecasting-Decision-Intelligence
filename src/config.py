"""Central configuration: paths, reproducible settings, logging."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DEFAULT_CSV_NAME = "Auto_Sales_data.csv"

# Customer contact details are not needed for any analysis; excluded from the processed file.
PII_COLUMNS = ("phone", "address_line1", "contact_last_name", "contact_first_name")
RANDOM_SEED = 42

TARGET_LABELS = {
    "revenue": "Revenue (sum of SALES)",
    "units": "Units (sum of QUANTITYORDERED)",
    "orders": "Orders (distinct order numbers)",
}


@dataclass(frozen=True)
class ForecastSettings:
    """Reproducible forecasting settings. Frozen => hashable => safe cache key."""

    target: str = "revenue"                  # revenue | units | orders
    exclude_statuses: tuple[str, ...] = ("Cancelled",)
    freq: str = "MS"                         # monthly (see diagnostics.frequency_check for justification)
    test_periods: int = 3                    # final hold-out, untouched until model selection is done
    val_horizon: int = 3                     # rolling-origin forecast horizon
    min_train_periods: int = 18              # first rolling-origin training window
    primary_metric: str = "wape"             # wape | rmse | mae
    parsimony_tolerance: float = 0.05        # prefer simpler model if within 5% of best primary metric
    seasonal_period: int = 12
    min_seasonal_strength: float = 0.30      # evidence gate for seasonal models
    alpha: float = 0.05                      # 95% prediction intervals
    max_failed_fold_share: float = 0.0       # candidates failing any validation fold are ineligible (fair comparison)

    def to_dict(self) -> dict:
        return asdict(self)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once (and a file under ./logs if writable)."""
    root = logging.getLogger()
    if root.handlers:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "autocast.log"))
    except OSError:  # read-only deployments
        pass
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def get_groq_key() -> str | None:
    """Return the Groq key from env/.env (never logged, never hardcoded)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key or None
