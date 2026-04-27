from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import hybrid_main  # noqa: E402


def main():
    hybrid_main()


if __name__ == "__main__":
    main()

