import subprocess
import sys

def main() -> None:
    subprocess.run(
            [sys.executable, r"C:\Users\Brian\Documents\Universiteit\Evolutionary_computing\EvolutionaryComputing2026\assignments\assignment_2\EA_1.py"],
            check=True,
        )
    
    subprocess.run(
        [sys.executable, r"C:\Users\Brian\Documents\Universiteit\Evolutionary_computing\EvolutionaryComputing2026\assignments\assignment_2\EA_2.py"],
        check=True,
    )

    subprocess.run(
        [sys.executable, r"C:\Users\Brian\Documents\Universiteit\Evolutionary_computing\EvolutionaryComputing2026\assignments\assignment_2\EA_3.py"],
        check=True,
    )

    subprocess.run(
        [sys.executable, r"C:\Users\Brian\Documents\Universiteit\Evolutionary_computing\EvolutionaryComputing2026\assignments\assignment_2\A2_2026_random_search.py"],
        check=True,
    )

if __name__ == "__main__":
    main()