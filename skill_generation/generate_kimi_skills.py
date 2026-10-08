"""Backward-compatible import for the ALFWorld skill generator."""

try:
    from .generate_skills_alfworld import *
    from .generate_skills_alfworld import main
except ImportError:  # direct execution from a source checkout
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skill_generation.generate_skills_alfworld import *
    from skill_generation.generate_skills_alfworld import main


if __name__ == "__main__":
    main()
