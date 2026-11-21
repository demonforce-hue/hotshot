"""Hotshots AI Sports Engine: Settlement, fractional Kelly ROI, and P&L chart for Supabase."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import requests


SIDE_TO_PROB_COL = {
    "H": "prob_home",
    "D": "prob_draw",
    "A": "prob_away",
}

SIDE_TO_ODDS_COL = {
    "H": "b365_h",
    "D": "b365_d",
    "A": "b365_a",
}


def _resolve_side_and_odds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Resolve bet side, model probability for that side, and decimal odds."""
    side_col = "best_value_side" if "best_value_side" in df.columns else "predicted_outcome"
    sides = df[side_col].astype(str).str.strip().str.upper()

    probs = []
    odds_vals = []

    has_best_odds = "best_odds" in df.columns
    for idx, side in sides.items():
        p_col = SIDE_TO_PROB_COL.get(side)
        probs.append(pd.to_numeric(df.loc[idx, p_col], errors="coerce") if p_col and p_col in df.columns else np.nan)

        if has_best_odds and pd.notna(pd.to_numeric(df.loc[idx, "best_odds"], errors="coerce")):
            odds_vals.append(pd.to_numeric(df.loc[idx, "best_odds"], errors="coerce"))
        else:
            o_col = SIDE_TO_ODDS_COL.get(side)
            odds_vals.append(pd.to_numeric(df.loc[idx, o_col], errors="coerce") if o_col and o_col in df.columns else np.nan)

    return (
        sides,
        pd.Series(probs, index=df.index, dtype="float64"),
        pd.Series(odds_vals, index=df.index, dtype="float64"),
    )


def compute_kelly_stakes(
    df: pd.DataFrame,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.04,
    bankroll: float = 1000.0,
) -> pd.DataFrame:
    """Compute fractional Kelly stake % and currency stake from model probability and odds."""
    out = df.copy()
    if out.empty:
        for col in ["bet_side", "selected_prob", "bet_odds", "kelly_stake_pct", "stake_amount"]:
            out[col] = pd.Series(dtype="float64")
        return out

    sides, p_raw, odds = _resolve_side_and_odds(out)
    p = p_raw.clip(1e-6, 1.0 - 1e-6)
    q = 1.0 - p
    b = odds - 1.0

    valid = (b > 0) & p.notna() & odds.notna()
    raw_kelly = np.where(valid, ((b * p) - q) / b.where(b > 0, np.nan), 0.0)
    raw_kelly = np.clip(raw_kelly, 0.0, None)

    capped_pct = np.minimum(raw_kelly * float(kelly_fraction), float(max_stake_pct))
    stake_amt = np.round(capped_pct * float(bankroll), 2)

    out["bet_side"] = sides
    out["selected_prob"] = np.round(p, 4)
    out["bet_odds"] = np.round(odds, 3)
    out["kelly_stake_pct"] = np.round(capped_pct, 4)
    out["stake_amount"] = stake_amt
    return out


def compute_brier_score(df: pd.DataFrame, actual_col: str = "actual_outcome") -> pd.Series:
    """3-class Brier score against actual_outcome ('H', 'D', 'A')."""
    y_h = (df[actual_col] == "H").astype(float)
    y_d = (df[actual_col] == "D").astype(float)
    y_a = (df[actual_col] == "A").astype(float)

    p_h = pd.to_numeric(df.get("prob_home"), errors="coerce")
    p_d = pd.to_numeric(df.get("prob_draw"), errors="coerce")
    p_a = pd.to_numeric(df.get("prob_away"), errors="coerce")

    score = ((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2) / 3.0
    valid = df[actual_col].isin(["H", "D", "A"]) & p_h.notna() & p_d.notna() & p_a.notna()
    return pd.Series(np.where(valid, np.round(score, 4), np.nan), index=df.index)


def settle_from_matches(
    predictions_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.04,
) -> pd.DataFrame:
    """Join football_predictions to football_matches on match_id and settle P&L + cumulative ROI."""
    preds = predictions_df.copy()
    if preds.empty:
        return preds

    matches = matches_df.copy()
    match_cols = [
        c
        for c in ["match_id", "result", "home_goals", "away_goals", "b365_h", "b365_d", "b365_a"]
        if c in matches.columns
    ]
    matches_clean = matches[match_cols].drop_duplicates(subset=["match_id"], keep="last").copy()
    matches_clean["match_id"] = matches_clean["match_id"].astype(str).str.strip()
    preds["match_id"] = preds["match_id"].astype(str).str.strip()

    merged = preds.merge(matches_clean, on="match_id", how="left", suffixes=("", "_match"))

    # Fill missing odds from football_matches if present
    for side, col in SIDE_TO_ODDS_COL.items():
        match_col = f"{col}_match"
        if match_col in merged.columns:
            merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(
                pd.to_numeric(merged[match_col], errors="coerce")
            )
            merged.drop(columns=[match_col], inplace=True)

    merged = compute_kelly_stakes(
        merged,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        bankroll=starting_bankroll,
    )

    actual = merged.get("result", pd.Series(index=merged.index, dtype="object")).astype(str).str.strip().str.upper()
    valid_actual = actual.isin(["H", "D", "A"])

    existing_actual = merged.get("actual_outcome", pd.Series(index=merged.index, dtype="object"))
    merged["actual_outcome"] = np.where(valid_actual, actual, existing_actual)
    merged["brier_score"] = compute_brier_score(merged, actual_col="actual_outcome")

    stake = pd.to_numeric(merged["stake_amount"], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(merged["bet_odds"], errors="coerce")
    side = merged["bet_side"]

    is_settled = merged["actual_outcome"].isin(["H", "D", "A"]) & (stake > 0) & odds.gt(1.0)
    is_win = is_settled & (side == merged["actual_outcome"])

    pnl = np.where(
        ~is_settled,
        np.nan,
        np.where(is_win, stake * (odds - 1.0), -stake),
    )

    merged["is_settled"] = is_settled
    merged["is_win"] = np.where(is_settled, is_win, np.nan)
    merged["pnl"] = np.round(pd.to_numeric(pnl, errors="coerce"), 2)

    merged = merged.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    merged["cum_pnl"] = np.round(merged["pnl"].fillna(0.0).cumsum(), 2)
    merged["bankroll_after"] = np.round(float(starting_bankroll) + merged["cum_pnl"], 2)

    cum_stake = np.where(merged["is_settled"], stake, 0.0).cumsum()
    merged["cum_roi_pct"] = np.where(
        cum_stake > 0,
        np.round((merged["cum_pnl"] / cum_stake) * 100.0, 2),
        0.0,
    )
    return merged


def fetch_table(
    table_name: str,
    select: str = "*",
    order_by: str = "match_date.asc",
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> pd.DataFrame:
    """Fetch rows from Supabase PostgREST."""
    url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
    key = supabase_key or os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    resp = requests.get(
        f"{url}/rest/v1/{table_name}?select={select}&order={order_by}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def upsert_predictions(
    df: pd.DataFrame,
    table_name: str = "football_predictions",
    on_conflict: Iterable[str] = ("match_id",),
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> dict:
    """Upsert settled prediction rows back into Supabase."""
    url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
    key = supabase_key or os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    if df.empty:
        return {"upserted": 0}

    # Keep only columns that exist on football_predictions (plus computed helper columns if added in DB)
    allowed_cols = [
        "match_id",
        "league_code",
        "match_date",
        "home_team",
        "away_team",
        "prob_home",
        "prob_draw",
        "prob_away",
        "implied_prob_home",
        "implied_prob_draw",
        "implied_prob_away",
        "edge_home",
        "edge_draw",
        "edge_away",
        "predicted_outcome",
        "actual_outcome",
        "brier_score",
    ]
    payload_df = df[[c for c in allowed_cols if c in df.columns]].copy()

    for col in payload_df.columns:
        if pd.api.types.is_datetime64_any_dtype(payload_df[col]):
            payload_df[col] = payload_df[col].dt.strftime("%Y-%m-%d")

    records = payload_df.where(pd.notnull(payload_df), None).to_dict(orient="records")
    conflict_str = ",".join(on_conflict)
    resp = requests.post(
        f"{url}/rest/v1/{table_name}?on_conflict={conflict_str}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json=records,
        timeout=30,
    )
    resp.raise_for_status()
    return {"upserted": len(records), "status_code": resp.status_code}


def render_pnl_panel(settled_df: pd.DataFrame):
    """Render ROI metrics + cumulative Profit/Loss chart in Streamlit."""
    import streamlit as st

    done = settled_df[settled_df["is_settled"] == True].copy() if "is_settled" in settled_df.columns else pd.DataFrame()
    if done.empty:
        st.info("No settled matches yet—Profit/Loss curve will appear once `football_matches.result` is populated.")
        return

    total_staked = float(pd.to_numeric(done["stake_amount"], errors="coerce").fillna(0.0).sum())
    total_pnl = float(pd.to_numeric(done["pnl"], errors="coerce").fillna(0.0).sum())
    roi_pct = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0
    win_rate = float(pd.to_numeric(done["is_win"], errors="coerce").fillna(0.0).mean() * 100.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROI", f"{roi_pct:+.1f}%")
    c2.metric("Profit / Loss", f"{total_pnl:+.2f}")
    c3.metric("Win Rate", f"{win_rate:.1f}%")

    chart_df = done[["match_date", "cum_pnl"]].copy()
    chart_df["match_date"] = pd.to_datetime(chart_df["match_date"])
    chart_df = chart_df.groupby("match_date", as_index=True)["cum_pnl"].last()
    st.line_chart(chart_df, height=230)
