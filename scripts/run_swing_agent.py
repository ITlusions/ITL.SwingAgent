import argparse
from swing_agent.agent import SwingAgent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ASML.AS")
    ap.add_argument("--interval", default="30m", choices=["15m","30m","1h","1d"])
    ap.add_argument("--lookback-days", type=int, default=30)
    ap.add_argument("--db", default=None, help="signals sqlite path")
    ap.add_argument("--vec-db", default=None, help="vector sqlite path")
    ap.add_argument("--sector", default="XLK", help="Sector ETF for relative strength (fallback SPY)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-llm-extras", action="store_true")
    args = ap.parse_args()

    agent = SwingAgent(interval=args.interval, lookback_days=args.lookback_days, log_db=args.db, vec_db=args.vec_db, use_llm=not args.no_llm, llm_extras=not args.no_llm_extras, sector_symbol=args.sector)
    sig = agent.analyze(args.symbol)
    print(sig.model_dump_json(indent=2))

    s = sig.model_dump()
    if s.get("action_plan"):
        print("\n--- ACTION PLAN ---")
        print(s["action_plan"])
    if s.get("risk_notes"):
        print("\n--- RISKS / INVALIDATION ---")
        print(s["risk_notes"])
    if s.get("scenarios"):
        print("\n--- SCENARIOS ---")
        for sc in s["scenarios"]:
            print(f"- {sc}")

if __name__ == "__main__":
    main()
