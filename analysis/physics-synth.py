"""
analysis/physics-synth.py
─────────────────────────
CLI entry point for physics_synth — single-note synthesis + comparison.
(physics_synth.py stays as the importable module; this file is the CLI face.)

Usage:
    python analysis/physics-synth.py --params analysis/params-ks-grand.json \
                                     --midi 60 --vel 3 --duration 6
    python analysis/physics-synth.py --preview
    python analysis/physics-synth.py --compare --midi 45 --vel 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.physics_synth import main

if __name__ == "__main__":
    main()
