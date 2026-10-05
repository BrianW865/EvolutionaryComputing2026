import json
from pathlib import Path
import numpy as np

CWD = Path.cwd()

files = {
    "EA1 (1 mutation)":  CWD / "__data__" / "EA_1" / "histories_variant1.json",
    "EA2 (3 mutations)": CWD / "__data__" / "EA_2" / "histories_variant2.json",
    "EA3 (5 mutations)": CWD / "__data__" / "EA_3" / "histories_variant3.json",
    "Random search":     CWD / "__data__" / "A2_2026_random_search" / "histories_random_search.json",
}

for name, path in files.items():
    with open(path) as f:
        histories = json.load(f)

    finals = [h[-1] for h in histories]        # final best fitness per seed
    lengths = [len(h) for h in histories]      # generations per seed

    print(f"{name}")
    print(f"  final fitness per seed : {np.round(finals, 3)}")
    print(f"  mean +- std (ddof=1)   : {np.mean(finals):.3f} +- {np.std(finals, ddof=1):.3f}")
    print(f"  generations per seed   : {lengths}")
    print(f"  generations min-max    : {min(lengths)}-{max(lengths)}  (mean {np.mean(lengths):.1f})")