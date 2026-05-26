#!/usr/bin/env python3
"""Aggregate individual sample_*.json files into web/samples.json for the demo viewer."""
import json
import sys
from pathlib import Path

def main(input_dir, output_path):
    inputs = sorted(Path(input_dir).glob("sample_*.json"))
    samples = []
    for p in inputs:
        with open(p) as f: samples.append(json.load(f))
        print(f"  loaded {p.name}: AR {samples[-1]['ar']['tps']:.1f} t/s, "
              f"SSD {samples[-1]['ssd']['tps']:.1f} t/s, sp {samples[-1]['speedup']:.2f}×")
    with open(output_path, "w") as f: json.dump(samples, f, indent=2)
    print(f"\n[aggregated {len(samples)} samples → {output_path}]")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    output = sys.argv[2] if len(sys.argv) > 2 else "web/samples.json"
    main(input_dir, output)
