import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["EA_1.py", "EA_2.py", "EA_3.py", "A2_2026_random_search.py"]

def main() -> None:
    for script in SCRIPTS:
        print(f"=== running {script} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)

if __name__ == "__main__":
    main()