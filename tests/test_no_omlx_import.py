import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def test_no_omlx_import():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_omlx_import.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
