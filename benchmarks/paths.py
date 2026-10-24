"""External model repositories, configurable through BGEONET_THIRD_PARTY."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=Path(os.environ.get('BGEONET_THIRD_PARTY',ROOT/'third_party')).resolve()
