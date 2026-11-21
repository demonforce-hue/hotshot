"""Fractional Kelly settlement, ROI metrics, and cumulative P&L chart for Hotshots AI Sports Engine."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


_SIDE_TO_PROB = {
    "H": "prob_home",
    "D": "prob_draw",
    "A": "prob_away",
}

_SIDE_TO_ODDS_CANDIDATES = {
    "H": ["best_odds", "B365H", "b365_h"],
    "D": ["best_odds", "B365D", "b365_d"],
    "A": ["best_odds", "B365A", "b365_a"],
}


def _resolve_bet_side(row: pd.Series) -> Optional[str]:
    for col in ("best_value_side", "predicted_outcome"):
        val = row.get(col)
        if pd.notna(val):
            s = str(val).strip().upper()
            if s in ("H", "D", "A"):
                return s

    edge_map = {
        "H": pd.to_numeric(row.get("edge_home"), errors="coerce"),
        "D": pd.to_numeric(row.get("edge_draw"), errors="coerce"),
        "A": pd.to_numeric(row.get("edge_away"), errors="coerce"),
    }
    valid_edges = {k: float(v) for k, v in edge_map.items() if pd.notna(v)}
    if valid_edges:
        return max(valid_edges.items(), key=lambda kv: kv[1])[0]
    return None


def _resolve_odds_for_side(row: pd.Series, side: Optional[str]) -> float:
    if side is None:
        return float("nan")

    # Prefer best_odds only when best_value_side matches or best_value_side isn't specified
    best_side = str(row.get("best_value_side", "")).strip().upper()
    for col in _SIDE_TO_ODDS_CANDIDATES.get(side, []):
        if col == "best_odds" and best_side and best_side != side:
            continue
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(val) and float(val) > 1.0:
            return float(val)
    return float("nan")


def attach_kelly_stakes(
    df: pd.DataFrame,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.04,
    bankroll: float = 1000.0,
    min_edge: float = 0.03,
) -> pd.DataFrame:
    """Attach kelly_side, kelly_prob, kelly_odds, kelly_stake_pct, and stake_amount."""
    out = df.copy()
    if out.empty:
        for col in ["kelly_side", "kelly_prob", "kelly_odds", "kelly_stake_pct", "stake_amount"]:
            out[col] = pd.Series(dtype="float64")
        return out

    sides: list[Optional[str]] = []
    probs: list[float] = []
    odds_list: list[float] = []
    edges: list[float] = []

    for _, row in out.iterrows():
        side = _resolve_bet_side(row)
        sides.append(side)

        p_col = _SIDE_TO_PROB.get(side or "", "")
        p_val = pd.to_numeric(row.get(p_col), errors="coerce") if p_col else np.nan
        probs.append(float(p_val) if pd.notna(p_val) else np.nan)

        o_val = _resolve_odds_for_side(row, side)
        odds_list.append(o_val)

        e_col = f"edge_{side.lower()}" if side else ""
        e_val = pd.to_numeric(row.get("best_edge"), errors="coerce")
        if pd.isna(e_val) and e_col:
            e_val = pd.to_numeric(row.get(e_col), errors="coerce")
        edges.append(float(e_val) if pd.notna(e_val) else np.nan)

    p = pd.Series(probs, index=out.index, dtype="float64").clip(1e-6, 1.0 - 1e-6)
    q = 1.0 - p
    odds = pd.Series(odds_list, index=out.index, dtype="float64")
    edge_s = pd.Series(edges, index=out.index, dtype="float64")
    b = odds - 1.0

    qualifies = (
        pd.Series(sides, index=out.index).isin(["H", "D", "A"])
        & (b > 0)
        & p.notna()
        & odds.notna()
        & (edge_s.isna() | (edge_s >= float(min_edge)))
    )

    raw_kelly = np.where(qualifies, np.maximum(0.0, ((b * p) - q) / b.where(b > 0, np.nan)), 0.0)
    stake_pct = np.where(
        qualifies,
        np.minimum(raw_kelly * float(kelly_fraction), float(max_stake_pct)),
        0.0,
    )
    stake_amt = np.round(stake_pct * float(bankroll), 2)

    out["kelly_side"] = sides
    out["kelly_prob"] = np.round(p, 4)
    out["kelly_odds"] = np.round(odds, 3)
    out["kelly_stake_pct"] = np.round(stake_pct, 4)
    out["stake_amount"] = stake_amt
    return out


def settle_predictions_with_kelly(
    df: pd.DataFrame,
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.04,
    min_edge: float = 0.03,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Settle rows with known outcomes ('H', 'D', 'A') using fractional Kelly stakes and return (df, stats)."""
    if df.empty:
        return df.copy(), {
            "bets": 0.0,
            "wins": 0.0,
            "win_rate": 0.0,
            "total_staked": 0.0,
            "pnl": 0.0,
            "roi": 0.0,
            "final_bankroll": float(starting_bankroll),
        }

    work = attach_kelly_stakes(
        df,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        bankroll=starting_bankroll,
        min_edge=min_edge,
    )

    outcome_col = "actual_outcome" if "actual_outcome" in work.columns else "result"
    actual = (
        work.get(outcome_col, pd.Series(index=work.index, dtype="object"))
        .astype(str)
        .str.strip()
        .str.upper()
    )
    stake = pd.to_numeric(work["stake_amount"], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(work["kelly_odds"], errors="coerce")
    side = work["kelly_side"].astype(str).str.strip().str.upper()

    is_bet = actual.isin(["H", "D", "A"]) & side.isin(["H", "D", "A"]) & (stake > 0) & odds.gt(1.0)
    is_win = is_bet & (side == actual)

    pnl = np.where(~is_bet, np.nan, np.where(is_win, stake * (odds - 1.0), -stake))
    work["is_settled_bet"] = is_bet
    work["is_win"] = np.where(is_bet, is_win, np.nan)
    work["pnl"] = np.round(pd.to_numeric(pnl, errors="coerce"), 2)

    sort_cols = [c for c in ["match_date", "match_id"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)

    work["cum_pnl"] = np.round(work["pnl"].fillna(0.0).cumsum(), 2)
    work["bankroll_after"] = np.round(float(starting_bankroll) + work["cum_pnl"], 2)

    cum_staked = np.where(work["is_settled_bet"], stake, 0.0).cumsum()
    work["cum_roi"] = np.where(cum_staked > 0, np.round(work["cum_pnl"] / cum_staked, 4), 0.0)

    settled_only = work[work["is_settled_bet"]]
    bets = float(len(settled_only))
    wins = float(settled_only["is_win"].fillna(False).sum()) if bets > 0 else 0.0
    total_staked = float(settled_only["stake_amount"].sum()) if bets > 0 else 0.0
    total_pnl = float(settled_only["pnl"].sum()) if bets > 0 else 0.0
    win_rate = (wins / bets) if bets > 0 else 0.0
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

    stats = {
        "bets": bets,
        "wins": wins,
        "win_rate": float(win_rate),
        "total_staked": round(total_staked, 2),
        "pnl": round(total_pnl, 2),
        "roi": float(roi),
        "final_bankroll": round(float(starting_bankroll) + total_pnl, 2),
    }
    return work, stats


def fetch_settled_from_supabase(url: str, headers: Dict[str, str], leagues: Optional[list[str]] = None) -> pd.DataFrame:
    """Fetch predictions joined with match results/odds from Supabase PostgREST."""
    if not url or not headers:
        return pd.DataFrame()
    try:
        r_preds = requests.get(
            f"{url}/rest/v1/football_predictions?select=*&order=match_date.asc",
            headers=headers,
            timeout=20,
        )
        if not r_preds.ok:
            return pd.DataFrame()
        preds = pd.DataFrame(r_preds.json())
        if preds.empty:
            return pd.DataFrame()

        r_matches = requests.get(
            f"{url}/rest/v1/football_matches?select=match_id,result,home_goals,away_goals,b365_h,b365_d,b365_a",
            headers=headers,
            timeout=20,
        )
        if r_matches.ok:
            matches = pd.DataFrame(r_matches.json())
            if not matches.empty and "match_id" in matches.columns:
                preds = preds.merge(matches.drop_duplicates(subset=["match_id"]), on="match_id", how="left")

        if leagues and "league_code" in preds.columns:
            preds = preds[preds["league_code"].isin(leagues)].copy()
        return preds
    except Exception:
        return pd.DataFrame()


def render_kelly_pnl_chart(settled_df: pd.DataFrame, stats: Dict[str, float], height: int = 250) -> None:
    """Render Kelly ROI summary metrics and cumulative Profit/Loss chart in Streamlit."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelly ROI", f"{stats['roi']:+.2%}")
    c2.metric("Net Profit / Loss", f"{stats['pnl']:+.2f}")
    c3.metric("Win Rate", f"{stats['win_rate']:.1%}")
    c4.metric("Bankroll", f"{stats['final_bankroll']:.2f}")

    if settled_df.empty or "is_settled_bet" not in settled_df.columns:
        st.info("No settled Kelly bets available yet.")
        return

    done = settled_df[settled_df["is_settled_bet"] == True].copy()
    if done.empty:
        st.info("No settled Kelly bets available yet.")
        return

    if "match_date" in done.columns:
        chart_df = done[["match_date", "cum_pnl"]].copy()
        chart_df["match_date"] = pd.to_datetime(chart_df["match_date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["match_date"]).groupby("match_date", as_index=True)["cum_pnl"].last()
    else:
        chart_df = done[["cum_pnl"]].reset_index(drop=True)

    st.line_chart(chart_df, height=height)
