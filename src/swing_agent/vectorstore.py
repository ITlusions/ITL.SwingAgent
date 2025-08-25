from __future__ import annotations
import sqlite3, json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS vec_store (
  id TEXT PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  vec_json TEXT NOT NULL,
  realized_r REAL,
  exit_reason TEXT,
  payload_json TEXT
);
CREATE INDEX IF NOT EXISTS vec_store_symbol_idx ON vec_store(symbol);
"""

def _ensure_db(db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.executescript(SCHEMA)

def add_vector(db_path: str | Path, *, vid: str, ts_utc: str, symbol: str, timeframe: str, vec: np.ndarray, realized_r: float | None, exit_reason: str | None, payload: Dict[str, Any] | None):
    db = Path(db_path); _ensure_db(db)
    v = vec.astype(float).tolist()
    with sqlite3.connect(db) as con:
        con.execute("INSERT OR REPLACE INTO vec_store (id, ts_utc, symbol, timeframe, vec_json, realized_r, exit_reason, payload_json) VALUES (?,?,?,?,?,?,?,?)", (vid, ts_utc, symbol, timeframe, json.dumps(v), realized_r, exit_reason, json.dumps(payload) if payload else None))

def update_vector_payload(db_path: str | Path, *, vid: str, merge: Dict[str, Any]):
    db = Path(db_path); _ensure_db(db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT payload_json FROM vec_store WHERE id=?", (vid,)).fetchone()
        payload = {}
        if row and row["payload_json"]:
            try: payload = json.loads(row["payload_json"])
            except Exception: payload = {}
        payload.update(merge)
        con.execute("UPDATE vec_store SET payload_json=? WHERE id=?", (json.dumps(payload), vid))

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
    if nu == 0 or nv == 0: return 0.0
    return float(np.dot(u, v) / (nu * nv))

def knn(db_path: str | Path, *, query_vec: np.ndarray, k: int = 50, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    db = Path(db_path); _ensure_db(db)
    rows = []
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT * FROM vec_store WHERE symbol=?;" if symbol else "SELECT * FROM vec_store;", (symbol,) if symbol else ())
        for r in cur.fetchall():
            vec = np.array(json.loads(r["vec_json"]), dtype=float)
            sim = cosine(query_vec, vec)
            rows.append({"id": r["id"], "ts_utc": r["ts_utc"], "symbol": r["symbol"], "timeframe": r["timeframe"], "similarity": sim, "realized_r": r["realized_r"], "exit_reason": r["exit_reason"], "payload": json.loads(r["payload_json"]) if r["payload_json"] else None})
    rows.sort(key=lambda x: x["similarity"], reverse=True)
    return rows[:k]

def filter_neighbors(neighbors: List[Dict[str, Any]], *, vol_regime: Optional[str] = None) -> List[Dict[str, Any]]:
    if not neighbors or not vol_regime: return neighbors
    filt = [n for n in neighbors if (n.get("payload") or {}).get("vol_regime") == vol_regime]
    return filt if len(filt) >= max(10, int(0.4*len(neighbors))) else neighbors

def extended_stats(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not neighbors:
        return {"n":0,"p_win":0.0,"avg_R":0.0,"avg_win_R":0.0,"avg_loss_R":0.0,"median_hold_bars":None,"median_hold_days":None,"median_win_hold_bars":None,"median_loss_hold_bars":None,"profit_factor":0.0,"tp":0,"sl":0,"time":0}
    rs = [x["realized_r"] for x in neighbors if x["realized_r"] is not None]
    wins = [r for r in rs if r > 0]; losses = [r for r in rs if r <= 0]
    p_win = (len(wins)/len(rs)) if rs else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_R = float(np.mean(rs)) if rs else 0.0
    profit_factor = (sum(wins)/abs(sum(losses))) if losses and sum(losses)!=0 else (float("inf") if sum(wins)>0 else 0.0)
    def collect(key):
        vals=[]; 
        for x in neighbors:
            v=(x.get("payload") or {}).get(key)
            if v is not None: vals.append(float(v))
        return vals
    all_b = collect("hold_bars")
    win_b=[]; loss_b=[]
    for x in neighbors:
        hb=(x.get("payload") or {}).get("hold_bars"); r=x.get("realized_r")
        if hb is None or r is None: continue
        (win_b if r>0 else loss_b).append(float(hb))
    import numpy as np
    med_all = int(np.median(all_b)) if all_b else None
    med_win = int(np.median(win_b)) if win_b else None
    med_loss = int(np.median(loss_b)) if loss_b else None
    bars_to_days = lambda b: None if b is None else round(b/13.0,2)
    return {"n":len(neighbors),"p_win":round(p_win,3),"avg_R":round(avg_R,3),"avg_win_R":round(avg_win,3),"avg_loss_R":round(avg_loss,3),"median_hold_bars":med_all,"median_hold_days":bars_to_days(med_all),"median_win_hold_bars":med_win,"median_loss_hold_bars":med_loss,"profit_factor":(round(profit_factor,3) if profit_factor!=float('inf') else float('inf')),"tp":sum(1 for x in neighbors if x.get("exit_reason")=="TP"),"sl":sum(1 for x in neighbors if x.get("exit_reason")=="SL"),"time":sum(1 for x in neighbors if x.get("exit_reason")=="TIME")}
