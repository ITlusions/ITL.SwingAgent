"""Placeholder pre-market scan that lists symbols to analyze."""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default="watchlists/ai_watchlist.txt")
    args = ap.parse_args()
    with open(args.watchlist) as f:
        symbols = [ln.strip() for ln in f if ln.strip()]
    print(f"Pre-market scan for {len(symbols)} symbols: {', '.join(symbols)}")


if __name__ == "__main__":
    main()
