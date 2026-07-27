import importlib.resources
from pathlib import Path


class Config:
    """Configuration class for DeepSpotM model."""

    # Gene vocabulary. `files()` replaces `importlib.resources.path()`, which is
    # deprecated since Python 3.11 and hands back a context-managed path that is
    # released on exit, so the value could outlive the guarantee it was valid.
    try:
        ALPHABET_PATH = Path(
            str(importlib.resources.files("deepspotm.assets") / "tokens.csv")
        )
    except Exception:
        ALPHABET_PATH = Path(__file__).resolve().parent / "assets" / "tokens.csv"


config = Config()
