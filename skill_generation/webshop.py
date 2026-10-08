"""Backward-compatible import for the WebShop skill generator."""

try:
    from .generate_skills_webshop import *
    from .generate_skills_webshop import main
except ImportError:  # direct execution from a source checkout
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skill_generation.generate_skills_webshop import *
    from skill_generation.generate_skills_webshop import main


if __name__ == "__main__":
    main()
