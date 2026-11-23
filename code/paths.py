"""Where the release finds its data and the simulator repository.
DATA: this release's data/ folder (override with PHADJ_DATA).
REPO: the simulator and network code, by default the copy in code/simulator (same layout as the phinder-adjoint-optics
repository; set PHADJ_REPO to use a full checkout instead)."""
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PHADJ_DATA", RELEASE / "data"))
REPO = Path(os.environ.get("PHADJ_REPO", RELEASE / "code" / "simulator"))
