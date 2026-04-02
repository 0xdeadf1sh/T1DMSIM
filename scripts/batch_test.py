"""
Batch test: generate data for many seeds and print summary statistics.
Usage: python scripts/batch_test.py [--seeds N] [--hours H]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from simulator import T1DMSimulator


def main():
    parser = argparse.ArgumentParser(description='Batch test BG distributions')
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds to test')
    parser.add_argument('--hours', type=float, default=72, help='Hours to simulate per seed')
    args = parser.parse_args()

    print(f"Testing {args.seeds} seeds, {args.hours}h each")
    print(f"{'Seed':>5} {'Min':>5} {'Max':>5} {'Mean':>6} {'Std':>6} {'TIR%':>5} "
          f"{'TBR%':>5} {'TAR%':>5} {'s1':>5} {'s2':>5} {'s3':>5} {'s4':>5}")
    print("-" * 85)

    all_tir = []
    for seed in range(args.seeds):
        sim = T1DMSimulator(seed=seed)
        data = sim.generate_hours(args.hours)
        bg = data['bg']

        tir = np.mean((bg >= 70) & (bg <= 180)) * 100
        tbr = np.mean(bg < 70) * 100
        tar = np.mean(bg > 180) * 100
        all_tir.append(tir)

        p = sim.patient
        print(f"{seed:5d} {bg.min():5.0f} {bg.max():5.0f} {bg.mean():6.0f} {bg.std():6.0f} "
              f"{tir:5.1f} {tbr:5.1f} {tar:5.1f} "
              f"{p.dietary_discipline:5.2f} {p.attentiveness:5.2f} "
              f"{p.dosing_competence:5.2f} {p.lifestyle_consistency:5.2f}")

    print("-" * 85)
    print(f"TIR across patients: mean={np.mean(all_tir):.1f}%, "
          f"min={np.min(all_tir):.1f}%, max={np.max(all_tir):.1f}%, "
          f"std={np.std(all_tir):.1f}%")


if __name__ == '__main__':
    main()
