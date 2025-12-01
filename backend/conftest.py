import sys
from pathlib import Path

# Ensure repository root is importable for `import backend`
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
