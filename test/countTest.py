#!/usr/bin/env python3
"""
10-second counting benchmark.

Goal: see how many loop iterations your CPU + Python can do in ~10 seconds.
This is NOT a scientific benchmark (Python version/settings matter),
but it's a fun, consistent "who's faster" test.

Run:
  python count_benchmark.py
Options:
  python count_benchmark.py --seconds 10 --warmup 1
"""

import argparse
import sys
import time


def bench(seconds: float, warmup: float) -> None:
    # Warmup: let CPU scale up / caches settle a bit
    if warmup > 0:
        end_warm = time.perf_counter() + warmup
        x = 0
        while time.perf_counter() < end_warm:
            # light work
            x = (x * 1664525 + 1013904223) & 0xFFFFFFFF

    # Benchmark
    start = time.perf_counter()
    end = start + seconds

    i = 0
    x = 0x12345678  # state to prevent "empty loop" behavior
    while True:
        # Do some cheap integer ops so it's not just i += 1
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF  # LCG step
        i += 1

        # Check time occasionally to reduce timer-call overhead
        if (i & 0x3FFFF) == 0:  # every 262,144 iterations
            if time.perf_counter() >= end:
                break

    elapsed = time.perf_counter() - start
    ips = i / elapsed if elapsed > 0 else 0.0

    print("\n=== 10s Counting Benchmark ===")
    print(f"Python: {sys.version.split()[0]} ({sys.implementation.name})")
    print(f"Platform: {sys.platform}")
    print(f"Target seconds: {seconds}")
    print(f"Elapsed: {elapsed:.6f} s")
    print(f"Iterations: {i:,}")
    print(f"Iterations/sec: {ips:,.0f}")
    print(f"Final state (ignore): {x}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0, help="Benchmark duration in seconds.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup duration in seconds.")
    args = parser.parse_args()

    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    bench(args.seconds, args.warmup)


if __name__ == "__main__":
    main()