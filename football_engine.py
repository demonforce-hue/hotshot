import csv
import hashlib
import os
import re
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st

LEAGUE_NAMES: Dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

DEFAULT_SEASONS: List[str] = ["2122", "2223", "2324", "2425", "2526", "2627"]

FEATURE_COLS: List[str] = [
    "h_elo",
    "a_elo",
    "elo_diff",
    "elo_exp_h",
    "h_pts5",
    "a_pts5",
    "pts5_diff",
    "h_gd5",
    "a_gd5",
    "gd5_diff",
    "h_gf10",
    "h_ga10",
    "a_gf10",
    "a_ga10",
    "h_sh_diff5",
    "a_sh_diff5",
    "h_sot_diff5",
    "a_sot_diff5",
    "h_gp",
    "a_gp",
    "imp_prob_h",
    "imp_prob_d",
    "imp_prob_a",
]

_ROW_START_RE = re.compile(r"^(?:\ufeff|ï»¿)?(Div|League|[A-Z]{1,3}\d?),", re.IGNORECASE)

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


def _clean_col_name(name: str, idx: int) -> str:
    cleaned = str(name).replace("\ufeff", "").replace("ï»¿", "").strip()
    return cleaned if cleaned else f"col_{idx}"


def _get_secret_or_env(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def _supabase_headers() -> Tuple[str, Dict[str, str]]:
    url = _get_secret_or_env("SUPABASE_URL").rstrip("/")
    key = (
        _get_secret_or_env("SUPABASE_SERVICE_ROLE_KEY")
        or _get_secret_or_env("SUPABASE_KEY")
        or _get_secret_or_env("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        return "", {}
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    return url, headers


def _unwrap_csv_lines(csv_text: str) -> str:
    """Joins wrapped continuation lines and drops blank lines in fixtures.csv."""
    lines = csv_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stitched: List[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _ROW_START_RE.match(stripped) or not stitched:
            stitched.append(stripped)
        else:
            if not stitched[-1].endswith(",") and not stripped.startswith(","):
                stitched[-1] = f"{stitched[-1]} {stripped}"
            else:
                stitched[-1] = f"{stitched[-1]}{stripped}"
    return "\n".join(stitched)


def _read_csv_resilient(csv_text: str, header: Optional[int] = 0) -> pd.DataFrame:
    cleaned_text = _unwrap_csv_lines(csv_text)
    rows: List[List[str]] = []
    reader = csv.reader(StringIO(cleaned_text), skipinitialspace=True)
    for r in reader:
        if any(cell.strip() for cell in r):
            rows.append([cell.replace("\ufeff", "").replace("ï»¿", "").strip() for cell in r])
    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)
    padded = [r + [""] * (max_len - len(r)) for r in rows]

    if header == 0:
        headers = [_clean_col_name(h, idx) for idx, h in enumerate(padded[0])]
        return pd.DataFrame(padded[1:], columns=headers)
    return pd.DataFrame(padded)


def build_match_id(league_code: str, match_date: str, home_team: str, away_team: str) -> str:
    raw = f"{league_code}|{match_date}|{home_team.strip()}|{away_team.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def fetch_season_csv(season: str, league_code: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = _read_csv_resilient(resp.text, header=0)
        if df.empty or not {"HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
            return None
        df = df.copy()
        meta = pd.DataFrame(
            {"league_code": league_code, "season": season},
            index=df.index,
        )
        return pd.concat([df, meta], axis=1).copy()
    except Exception:
        return None


def load_english_football_data(
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    leagues = leagues or ["E0", "E1"]
    seasons = seasons or DEFAULT_SEASONS

    frames: List[pd.DataFrame] = []
    for season in seasons:
        for league in leagues:
            raw_df = fetch_season_csv(season, league)
            if raw_df is not None:
                frames.append(raw_df)

    if not frames:
        raise RuntimeError("No English football data could be loaded from football-data.co.uk.")

    df = pd.concat(frames, ignore_index=True).copy()

    parsed_dates = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, "Date"].astype(str).str.strip(), format="%d/%m/%y", errors="coerce"
        )

    numeric_cols: Dict[str, pd.Series] = {}
    for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"]:
        if col in df.columns:
            numeric_cols[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            numeric_cols[col] = pd.Series(np.nan, index=df.index, dtype="float64")

    home_team_s = df["HomeTeam"].astype(str).str.strip()
    away_team_s = df["AwayTeam"].astype(str).str.strip()
    match_date_s = parsed_dates.dt.strftime("%Y-%m-%d")

    result_s = df["FTR"].astype(str).str.strip() if "FTR" in df.columns else pd.Series(np.nan, index=df.index, dtype="object")
    result_s = result_s.where(result_s.isin(["H", "D", "A"]), np.nan)

    match_id_s = pd.Series(
        [
            build_match_id(l, d, h, a)
            for l, d, h, a in zip(df["league_code"], match_date_s, home_team_s, away_team_s)
        ],
        index=df.index,
        dtype="object",
    )

    clean_df = pd.concat(
        [
            df[["league_code", "season"]].reset_index(drop=True),
            pd.DataFrame(
                {
                    "match_id": match_id_s.reset_index(drop=True),
                    "match_date_dt": parsed_dates.reset_index(drop=True),
                    "match_date": match_date_s.reset_index(drop=True),
                    "home_team": home_team_s.reset_index(drop=True),
                    "away_team": away_team_s.reset_index(drop=True),
                    "home_goals": numeric_cols["FTHG"].reset_index(drop=True),
                    "away_goals": numeric_cols["FTAG"].reset_index(drop=True),
                    "result": result_s.reset_index(drop=True),
                    "HS": numeric_cols["HS"].reset_index(drop=True),
                    "AS": numeric_cols["AS"].reset_index(drop=True),
                    "HST": numeric_cols["HST"].reset_index(drop=True),
                    "AST": numeric_cols["AST"].reset_index(drop=True),
                    "B365H": numeric_cols["B365H"].reset_index(drop=True),
                    "B365D": numeric_cols["B365D"].reset_index(drop=True),
                    "B365A": numeric_cols["B365A"].reset_index(drop=True),
                }
            ),
        ],
        axis=1,
    )

    clean_df = clean_df.dropna(subset=["match_date_dt", "home_team", "away_team"]).copy()
    clean_df = clean_df.sort_values(["match_date_dt", "league_code", "match_id"]).reset_index(drop=True)
    clean_df = clean_df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    return clean_df.copy()


def _extract_odds_from_row(row: pd.Series) -> Tuple[float, float, float]:
    for h_col, d_col, a_col in [
        ("B365H", "B365D", "B365A"),
        ("BbMxH", "BbMxD", "BbMxA"),
        ("AvgH", "AvgD", "AvgA"),
    ]:
        if h_col in row.index and d_col in row.index and a_col in row.index:
            oh = pd.to_numeric(row[h_col], errors="coerce")
            od = pd.to_numeric(row[d_col], errors="coerce")
            oa = pd.to_numeric(row[a_col], errors="coerce")
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa) and oh > 1.0 and od > 1.0 and oa > 1.0:
                return float(oh), float(od), float(oa)

    vals: List[float] = []
    for v in row.iloc[5:]:
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num) and 1.01 <= float(num) <= 50.0:
            vals.append(float(num))
        elif vals and len(vals) % 3 != 0:
            vals.clear()

    if len(vals) >= 3:
        return vals[0], vals[1], vals[2]
    return np.nan, np.nan, np.nan


def fetch_upcoming_fixtures(leagues: Optional[List[str]] = None, timeout: int = 20) -> Tuple[pd.DataFrame, str]:
    target_leagues = set(leagues or ["E0", "E1", "E2", "E3"])
    url = f"https://www.football-data.co.uk/fixtures.csv?cb={int(time.time())}"

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return pd.DataFrame(), f"HTTP {resp.status_code} from fixtures.csv"
    except Exception as exc:
        return pd.DataFrame(), f"Fetch error: {exc}"

    raw_text = resp.text
    df_header = _read_csv_resilient(raw_text, header=0)
    has_named_cols = not df_header.empty and {"HomeTeam", "AwayTeam"}.issubset(set(df_header.columns))

    if has_named_cols:
        df = df_header.copy()
        div_col = next((c for c in ["Div", "League", "col_0"] if c in df.columns), None)
        date_col = next((c for c in ["Date", "col_1"] if c in df.columns), None)
        if div_col is None or date_col is None:
            return pd.DataFrame(), f"Missing division/date column. Columns: {list(df.columns[:8])}"
        df["league_code"] = df[div_col].astype(str).str.strip()
        df["date_raw"] = df[date_col].astype(str).str.strip()
        df["home_raw"] = df["HomeTeam"].astype(str).str.strip()
        df["away_raw"] = df["AwayTeam"].astype(str).str.strip()
    else:
        df = _read_csv_resilient(raw_text, header=None)
        if df.empty or df.shape[1] < 5:
            return pd.DataFrame(), f"fixtures.csv too short or empty (cols={df.shape[1] if not df.empty else 0})"
        df = df.copy()
        df["league_code"] = df.iloc[:, 0].astype(str).str.strip()
        df["date_raw"] = df.iloc[:, 1].astype(str).str.strip()
        df["home_raw"] = df.iloc[:, 3].astype(str).str.strip()
        df["away_raw"] = df.iloc[:, 4].astype(str).str.strip()

    df = df[df["league_code"].isin(target_leagues)].copy()
    if df.empty:
        return pd.DataFrame(), f"No rows in fixtures.csv for leagues {sorted(target_leagues)}"

    parsed_dates = pd.to_datetime(df["date_raw"], format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, "date_raw"], format="%d/%m/%y", errors="coerce"
        )

    match_date_s = parsed_dates.dt.strftime("%Y-%m-%d")
    season_code = DEFAULT_SEASONS[-1] if DEFAULT_SEASONS else "2627"

    b365_h_list: List[float] = []
    b365_d_list: List[float] = []
    b365_a_list: List[float] = []
    for _, row in df.iterrows():
        oh, od, oa = _extract_odds_from_row(row)
        b365_h_list.append(oh)
        b365_d_list.append(od)
        b365_a_list.append(oa)

    match_id_s = pd.Series(
        [
            build_match_id(l, d, h, a)
            for l, d, h, a in zip(df["league_code"], match_date_s, df["home_raw"], df["away_raw"])
        ],
        index=df.index,
        dtype="object",
    )

    out = pd.DataFrame(
        {
            "league_code": df["league_code"].reset_index(drop=True),
            "season": pd.Series(season_code, index=range(len(df)), dtype="object"),
            "match_id": match_id_s.reset_index(drop=True),
            "match_date_dt": parsed_dates.reset_index(drop=True),
            "match_date": match_date_s.reset_index(drop=True),
            "home_team": df["home_raw"].reset_index(drop=True),
            "away_team": df["away_raw"].reset_index(drop=True),
            "home_goals": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "away_goals": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "result": pd.Series(np.nan, index=range(len(df)), dtype="object"),
            "HS": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "AS": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "HST": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "AST": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "B365H": pd.Series(b365_h_list, index=range(len(df)), dtype="float64"),
            "B365D": pd.Series(b365_d_list, index=range(len(df)), dtype="float64"),
            "B365A": pd.Series(b365_a_list, index=range(len(df)), dtype="float64"),
        }
    )

    out = out.dropna(subset=["match_date_dt", "home_team", "away_team"]).copy()
    out = out.sort_values(["match_date_dt", "league_code", "match_id"]).reset_index(drop=True)
    return out, f"Parsed {len(out)} upcoming fixtures from fixtures.csv"


def fetch_unplayed_from_supabase(leagues: Optional[List[str]] = None) -> pd.DataFrame:
    url, headers = _supabase_headers()
    if not url or not headers:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"{url}/rest/v1/football_predictions?actual_outcome=is.null&select=*",
            headers=headers,
            timeout=15,
        )
        if not resp.ok:
            return pd.DataFrame()
        rows = resp.json()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if leagues and "league_code" in df.columns:
            df = df[df["league_code"].isin(leagues)].copy()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_settled_from_supabase(leagues: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch predictions joined with football_matches actual results/odds from Supabase."""
    url, headers = _supabase_headers()
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
                preds["match_id"] = preds["match_id"].astype(str).str.strip()
                matches["match_id"] = matches["match_id"].astype(str).str.strip()
                preds = preds.merge(
                    matches.drop_duplicates(subset=["match_id"], keep="last"),
                    on="match_id",
                    how="left",
                    suffixes=("", "_match"),
                )

        if leagues and "league_code" in preds.columns:
            preds = preds[preds["league_code"].isin(leagues)].copy()
        return preds
    except Exception:
        return pd.DataFrame()


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

    sides: List[Optional[str]] = []
    probs: List[float] = []
    odds_list: List[float] = []
    edges: List[float] = []

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
    """Settle rows with known outcomes ('H', 'D', 'A') using fractional Kelly stakes."""
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

    # Prefer football_matches.result if merged, otherwise actual_outcome or result
    result_series = work.get("result", pd.Series(index=work.index, dtype="object"))
    actual_series = work.get("actual_outcome", pd.Series(index=work.index, dtype="object"))
    outcome = result_series.combine_first(actual_series).astype(str).str.strip().str.upper()
    work["actual_outcome"] = outcome.where(outcome.isin(["H", "D", "A"]), np.nan)

    stake = pd.to_numeric(work["stake_amount"], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(work["kelly_odds"], errors="coerce")
    side = work["kelly_side"].astype(str).str.strip().str.upper()

    is_bet = work["actual_outcome"].isin(["H", "D", "A"]) & side.isin(["H", "D", "A"]) & (stake > 0) & odds.gt(1.0)
    is_win = is_bet & (side == work["actual_outcome"])

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


def render_kelly_pnl_chart(settled_df: pd.DataFrame, stats: Dict[str, float], height: int = 250) -> None:
    """Render Kelly ROI KPIs and cumulative Profit/Loss chart in Streamlit."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelly ROI", f"{stats['roi']:+.2%}")
    c2.metric("Net P&L", f"{stats['pnl']:+.2f}")
    c3.metric("Kelly Win Rate", f"{stats['win_rate']:.1%}")
    c4.metric("Ending Bankroll", f"{stats['final_bankroll']:.2f}")

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


def engineer_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly pre-match features (Elo, rolling form, goal/shot differentials,
    and market-implied probabilities) so there is zero future data leakage.
    """
    data = df.sort_values(["match_date_dt", "match_id"]).reset_index(drop=True).copy()

    elo: Dict[str, float] = {}
    history: Dict[str, List[Dict[str, float]]] = {}

    base_elo = 1500.0
    k_factor = 22.0
    home_advantage_elo = 65.0

    feat_rows: List[Dict[str, float]] = []

    def get_team_stats(team: str, n: int = 5) -> Dict[str, float]:
        recs = history.get(team, [])
        if not recs:
            return {
                "pts_avg": 1.0,
                "gf_avg": 1.2,
                "ga_avg": 1.2,
                "gd_avg": 0.0,
                "sh_diff_avg": 0.0,
                "sot_diff_avg": 0.0,
                "games_played": 0.0,
            }
        recent = recs[-n:]
        return {
            "pts_avg": float(np.mean([r["pts"] for r in recent])),
            "gf_avg": float(np.mean([r["gf"] for r in recent])),
            "ga_avg": float(np.mean([r["ga"] for r in recent])),
            "gd_avg": float(np.mean([r["gf"] - r["ga"] for r in recent])),
            "sh_diff_avg": float(np.mean([r["sh_diff"] for r in recent])),
            "sot_diff_avg": float(np.mean([r["sot_diff"] for r in recent])),
            "games_played": float(len(recs)),
        }

    for _, row in data.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        h_elo = elo.get(h_team, base_elo)
        a_elo = elo.get(a_team, base_elo)

        h5 = get_team_stats(h_team, n=5)
        a5 = get_team_stats(a_team, n=5)
        h10 = get_team_stats(h_team, n=10)
        a10 = get_team_stats(a_team, n=10)

        oh, od, oa = row.get("B365H"), row.get("B365D"), row.get("B365A")
        if pd.notna(oh) and pd.notna(od) and pd.notna(oa) and oh > 1 and od > 1 and oa > 1:
            inv_h, inv_d, inv_a = 1.0 / oh, 1.0 / od, 1.0 / oa
            total_inv = inv_h + inv_d + inv_a
            imp_h, imp_d, imp_a = inv_h / total_inv, inv_d / total_inv, inv_a / total_inv
        else:
            imp_h = imp_d = imp_a = np.nan

        elo_diff = (h_elo + home_advantage_elo) - a_elo
        expected_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        feat_rows.append(
            {
                "h_elo": h_elo,
                "a_elo": a_elo,
                "elo_diff": elo_diff,
                "elo_exp_h": expected_h,
                "h_pts5": h5["pts_avg"],
                "a_pts5": a5["pts_avg"],
                "pts5_diff": h5["pts_avg"] - a5["pts_avg"],
                "h_gd5": h5["gd_avg"],
                "a_gd5": a5["gd_avg"],
                "gd5_diff": h5["gd_avg"] - a5["gd_avg"],
                "h_gf10": h10["gf_avg"],
                "h_ga10": h10["ga_avg"],
                "a_gf10": a10["gf_avg"],
                "a_ga10": a10["ga_avg"],
                "h_sh_diff5": h5["sh_diff_avg"],
                "a_sh_diff5": a5["sh_diff_avg"],
                "h_sot_diff5": h5["sot_diff_avg"],
                "a_sot_diff5": a5["sot_diff_avg"],
                "h_gp": h5["games_played"],
                "a_gp": a5["games_played"],
                "imp_prob_h": imp_h,
                "imp_prob_d": imp_d,
                "imp_prob_a": imp_a,
            }
        )

        res = row.get("result")
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        if res in ("H", "D", "A") and pd.notna(hg) and pd.notna(ag):
            actual_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
            actual_a = 1.0 - actual_h

            exp_a = 1.0 - expected_h
            goal_margin = abs(float(hg) - float(ag))
            mult = max(1.0, np.log1p(goal_margin))

            elo[h_team] = h_elo + k_factor * mult * (actual_h - expected_h)
            elo[a_team] = a_elo + k_factor * mult * (actual_a - exp_a)

            pts_h = 3.0 if res == "H" else (1.0 if res == "D" else 0.0)
            pts_a = 3.0 if res == "A" else (1.0 if res == "D" else 0.0)

            sh_diff = (
                float(row["HS"] - row["AS"])
                if pd.notna(row.get("HS")) and pd.notna(row.get("AS"))
                else 0.0
            )
            sot_diff = (
                float(row["HST"] - row["AST"])
                if pd.notna(row.get("HST")) and pd.notna(row.get("AST"))
                else 0.0
            )

            history.setdefault(h_team, []).append(
                {"pts": pts_h, "gf": float(hg), "ga": float(ag), "sh_diff": sh_diff, "sot_diff": sot_diff}
            )
            history.setdefault(a_team, []).append(
                {"pts": pts_a, "gf": float(ag), "ga": float(hg), "sh_diff": -sh_diff, "sot_diff": -sot_diff}
            )

    feat_df = pd.DataFrame(feat_rows, index=data.index)
    return pd.concat([data, feat_df], axis=1).copy()


def run_walk_forward_backtest(
    df_feat: pd.DataFrame,
    min_train_rows: int = 500,
) -> Tuple[pd.DataFrame, float]:
    label_map = {"H": 0, "D": 1, "A": 2}
    inv_map = {0: "H", 1: "D", 2: "A"}

    completed = df_feat[df_feat["result"].isin(label_map.keys())].copy().reset_index(drop=True)
    if len(completed) <= min_train_rows:
        raise ValueError("Not enough completed matches for walk-forward backtest.")

    y_all = completed["result"].map(label_map).astype(int)
    X_all = completed[FEATURE_COLS]

    oof_probs = np.full((len(completed), 3), np.nan)
    step = 200

    for start in range(min_train_rows, len(completed), step):
        end = min(start + step, len(completed))
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=23,
            subsample_for_bin=2000,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_all.iloc[:start], y_all.iloc[:start])
        oof_probs[start:end] = model.predict_proba(X_all.iloc[start:end])

    valid_mask = ~np.isnan(oof_probs[:, 0])
    eval_df = completed.loc[valid_mask].copy()
    probs = oof_probs[valid_mask]

    y_one_hot = np.eye(3)[y_all.loc[valid_mask].to_numpy()]
    brier_per_row = np.sum((probs - y_one_hot) ** 2, axis=1)
    brier = float(np.mean(brier_per_row))

    pred_cols = pd.DataFrame(
        {
            "prob_home": probs[:, 0],
            "prob_draw": probs[:, 1],
            "prob_away": probs[:, 2],
            "predicted_outcome": [inv_map[int(i)] for i in np.argmax(probs, axis=1)],
            "brier_score": brier_per_row,
            "edge_home": probs[:, 0] - eval_df["imp_prob_h"].to_numpy(),
            "edge_draw": probs[:, 1] - eval_df["imp_prob_d"].to_numpy(),
            "edge_away": probs[:, 2] - eval_df["imp_prob_a"].to_numpy(),
        },
        index=eval_df.index,
    )

    return pd.concat([eval_df, pred_cols], axis=1).copy(), brier


def compute_simulated_roi(eval_df: pd.DataFrame, min_edge: float = 0.03) -> Dict[str, float]:
    df = eval_df[eval_df["result"].isin(["H", "D", "A"])].copy()
    if df.empty:
        return {"bets": 0.0, "wins": 0.0, "win_rate": 0.0, "profit_units": 0.0, "roi": 0.0}

    candidates = [
        ("H", "edge_home", "B365H"),
        ("D", "edge_draw", "B365D"),
        ("A", "edge_away", "B365A"),
    ]

    bets = 0
    wins = 0
    profit = 0.0

    for _, r in df.iterrows():
        actual = r["result"]
        best_side = None
        best_edge = min_edge
        best_odds = np.nan

        for side, edge_col, odds_col in candidates:
            edge_val = r.get(edge_col)
            odds_val = r.get(odds_col)
            if pd.notna(edge_val) and pd.notna(odds_val) and float(odds_val) > 1.0:
                if float(edge_val) >= best_edge:
                    best_edge = float(edge_val)
                    best_side = side
                    best_odds = float(odds_val)

        if best_side is not None:
            bets += 1
            if actual == best_side:
                wins += 1
                profit += best_odds - 1.0
            else:
                profit -= 1.0

    win_rate = (wins / bets) if bets > 0 else 0.0
    roi = (profit / bets) if bets > 0 else 0.0
    return {
        "bets": float(bets),
        "wins": float(wins),
        "win_rate": float(win_rate),
        "profit_units": float(profit),
        "roi": float(roi),
    }


def score_upcoming_with_latest_model(
    df_feat: pd.DataFrame,
    upcoming_df: pd.DataFrame,
    min_train_rows: int = 500,
) -> pd.DataFrame:
    if upcoming_df.empty:
        return pd.DataFrame()

    label_map = {"H": 0, "D": 1, "A": 2}
    inv_map = {0: "H", 1: "D", 2: "A"}

    completed = df_feat[df_feat["result"].isin(label_map.keys())].copy()
    if len(completed) < min_train_rows:
        return pd.DataFrame()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=23,
        subsample_for_bin=2000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(completed[FEATURE_COLS], completed["result"].map(label_map).astype(int))

    up_feat = upcoming_df.copy()
    probs = model.predict_proba(up_feat[FEATURE_COLS])

    up_feat["prob_home"] = probs[:, 0]
    up_feat["prob_draw"] = probs[:, 1]
    up_feat["prob_away"] = probs[:, 2]
    up_feat["predicted_outcome"] = [inv_map[int(i)] for i in np.argmax(probs, axis=1)]
    up_feat["brier_score"] = np.nan

    best_sides: List[str] = []
    best_edges: List[float] = []
    best_odds_list: List[float] = []

    for side, p_col, odds_col, e_col in [
        ("H", "prob_home", "B365H", "edge_home"),
        ("D", "prob_draw", "B365D", "edge_draw"),
        ("A", "prob_away", "B365A", "edge_away"),
    ]:
        odds = pd.to_numeric(up_feat[odds_col], errors="coerce")
        imp = np.where(odds > 1.0, 1.0 / odds, np.nan)
        up_feat[f"imp_prob_{side.lower()}"] = imp
        up_feat[e_col] = up_feat[p_col] - imp

    for _, r in up_feat.iterrows():
        options = [
            ("H", r.get("edge_home"), r.get("B365H")),
            ("D", r.get("edge_draw"), r.get("B365D")),
            ("A", r.get("edge_away"), r.get("B365A")),
        ]
        valid_opts = [
            (s, float(e), float(o))
            for s, e, o in options
            if pd.notna(e) and pd.notna(o) and float(o) > 1.0
        ]
        if valid_opts:
            s_top, e_top, o_top = max(valid_opts, key=lambda x: x[1])
            best_sides.append(s_top)
            best_edges.append(e_top)
            best_odds_list.append(o_top)
        else:
            best_sides.append(r["predicted_outcome"])
            best_edges.append(np.nan)
            best_odds_list.append(np.nan)

    up_feat["best_value_side"] = best_sides
    up_feat["best_edge"] = best_edges
    up_feat["best_odds"] = best_odds_list
    return up_feat.copy()


def sync_predictions_to_supabase(eval_df: pd.DataFrame, limit: int = 250) -> int:
    url, headers = _supabase_headers()
    if not url or not headers:
        return 0

    recent = eval_df.tail(limit)

    match_payload = [
        {
            "match_id": r["match_id"],
            "league_code": r["league_code"],
            "season": r["season"],
            "match_date": r["match_date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_goals": None if pd.isna(r["home_goals"]) else int(r["home_goals"]),
            "away_goals": None if pd.isna(r["away_goals"]) else int(r["away_goals"]),
            "result": None if pd.isna(r["result"]) else r["result"],
            "b365_h": None if pd.isna(r["B365H"]) else float(r["B365H"]),
            "b365_d": None if pd.isna(r["B365D"]) else float(r["B365D"]),
            "b365_a": None if pd.isna(r["B365A"]) else float(r["B365A"]),
        }
        for _, r in recent.iterrows()
    ]

    pred_payload = [
        {
            "match_id": r["match_id"],
            "league_code": r["league_code"],
            "match_date": r["match_date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "prob_home": round(float(r["prob_home"]), 4),
            "prob_draw": round(float(r["prob_draw"]), 4),
            "prob_away": round(float(r["prob_away"]), 4),
            "implied_prob_home": None if pd.isna(r["imp_prob_h"]) else round(float(r["imp_prob_h"]), 4),
            "implied_prob_draw": None if pd.isna(r["imp_prob_d"]) else round(float(r["imp_prob_d"]), 4),
            "implied_prob_away": None if pd.isna(r["imp_prob_a"]) else round(float(r["imp_prob_a"]), 4),
            "edge_home": None if pd.isna(r["edge_home"]) else round(float(r["edge_home"]), 4),
            "edge_draw": None if pd.isna(r["edge_draw"]) else round(float(r["edge_draw"]), 4),
            "edge_away": None if pd.isna(r["edge_away"]) else round(float(r["edge_away"]), 4),
            "predicted_outcome": r["predicted_outcome"],
            "actual_outcome": None if pd.isna(r["result"]) else r["result"],
            "brier_score": None if pd.isna(r["brier_score"]) else round(float(r["brier_score"]), 4),
        }
        for _, r in recent.iterrows()
    ]

    r1 = requests.post(
        f"{url}/rest/v1/football_matches?on_conflict=match_id",
        headers=headers,
        json=match_payload,
        timeout=20,
    )
    if not r1.ok:
        raise RuntimeError(f"football_matches sync failed ({r1.status_code}): {r1.text}")

    r2 = requests.post(
        f"{url}/rest/v1/football_predictions?on_conflict=match_id",
        headers=headers,
        json=pred_payload,
        timeout=20,
    )
    if not r2.ok:
        raise RuntimeError(f"football_predictions sync failed ({r2.status_code}): {r2.text}")

    return len(pred_payload)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_pipeline(
    leagues_tuple: Tuple[str, ...],
    min_edge: float,
) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, float, float, Dict[str, float]]:
    raw = load_english_football_data(leagues=list(leagues_tuple))
    upcoming_raw, fixture_status = fetch_upcoming_fixtures(leagues=list(leagues_tuple))

    combined = raw.copy()
    if not upcoming_raw.empty:
        existing_ids = set(combined["match_id"].tolist())
        new_up = upcoming_raw[~upcoming_raw["match_id"].isin(existing_ids)].copy()
        if not new_up.empty:
            combined = pd.concat([combined, new_up], ignore_index=True).copy()

    feat = engineer_prematch_features(combined)
    backtest_df, brier = run_walk_forward_backtest(feat)
    acc = float((backtest_df["predicted_outcome"] == backtest_df["result"]).mean())
    roi_stats = compute_simulated_roi(backtest_df, min_edge=min_edge)

    unplayed_feat = feat[~feat["result"].isin(["H", "D", "A"])].copy()
    upcoming_scored = score_upcoming_with_latest_model(feat, unplayed_feat)
    sb_unplayed = fetch_unplayed_from_supabase(leagues=list(leagues_tuple))
    sb_settled = fetch_settled_from_supabase(leagues=list(leagues_tuple))

    return len(raw), backtest_df, upcoming_scored, sb_unplayed, sb_settled, fixture_status, acc, brier, roi_stats


def render_streamlit_app() -> None:
    st.set_page_config(page_title="Hotshots AI Sports Engine — English Football", layout="wide")
    st.title("Hotshots AI Sports Engine — English Football")
    st.caption("Walk-forward LightGBM + Elo + rolling form tracker with fractional Kelly ROI & P&L settlement")

    col_l, col_e, col_r, col_s = st.columns([2, 1, 1, 1])
    with col_l:
        selected_leagues = st.multiselect(
            "Leagues",
            options=["E0", "E1", "E2", "E3"],
            default=["E0", "E1", "E2", "E3"],
            format_func=lambda code: f"{code} — {LEAGUE_NAMES.get(code, code)}",
        )
    with col_e:
        min_edge_pct = st.slider("Min Edge (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
    with col_r:
        st.write("")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col_s:
        st.write("")
        sync_flag = st.checkbox("Sync to Supabase", value=False)

    if not selected_leagues:
        st.warning("Select at least one league to run the engine.")
        return

    min_edge = min_edge_pct / 100.0

    try:
        with st.spinner("Running Elo + form features, walk-forward backtest, and fixture edge scoring..."):
            total_matches, backtest_df, upcoming_scored, sb_unplayed, sb_settled, fixture_status, acc, brier, roi_stats = (
                _cached_pipeline(tuple(selected_leagues), min_edge)
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Walk-Forward Acc", f"{acc:.2%}")
        m2.metric("Brier Score", f"{brier:.4f}")
        m3.metric("Simulated ROI", f"{roi_stats['roi']:+.2%}")
        m4.metric("Net Units", f"{roi_stats['profit_units']:+.2f}u")
        m5.metric("Qualifying Bets", f"{int(roi_stats['bets']):,}")

        if sync_flag:
            to_sync = backtest_df.tail(250).copy()
            if not upcoming_scored.empty:
                to_sync = pd.concat([to_sync, upcoming_scored], ignore_index=True).drop_duplicates(
                    subset=["match_id"], keep="last"
                )
            synced = sync_predictions_to_supabase(to_sync, limit=len(to_sync))
            st.success(f"Synced {synced} rows to Supabase (out of {total_matches:,} loaded historical matches).")

        tab_edges, tab_roi, tab_history = st.tabs(
            ["Upcoming Fixtures & Edges", "ROI & League Breakdown", "Recent Backtest"]
        )

        with tab_edges:
            st.caption(f"Fixture source status: {fixture_status}")
            if not upcoming_scored.empty:
                value_df = upcoming_scored[
                    upcoming_scored["best_edge"].notna() & (upcoming_scored["best_edge"] >= min_edge)
                ].sort_values("best_edge", ascending=False)

                st.caption(
                    f"Showing {len(value_df)} upcoming fixtures with edge >= {min_edge_pct:.1f}% "
                    f"(out of {len(upcoming_scored)} total upcoming fixtures)."
                )
                edge_cols = [
                    "match_date",
                    "league_code",
                    "home_team",
                    "away_team",
                    "best_value_side",
                    "best_edge",
                    "best_odds",
                    "prob_home",
                    "prob_draw",
                    "prob_away",
                ]
                display_source = value_df if not value_df.empty else upcoming_scored.sort_values("match_date")
                st.dataframe(display_source[edge_cols], use_container_width=True)
            elif not sb_unplayed.empty:
                st.info("Showing unplayed forward predictions already stored in Supabase.")
                sb_cols = [
                    c
                    for c in [
                        "match_date",
                        "league_code",
                        "home_team",
                        "away_team",
                        "predicted_outcome",
                        "prob_home",
                        "prob_draw",
                        "prob_away",
                        "edge_home",
                        "edge_draw",
                        "edge_away",
                    ]
                    if c in sb_unplayed.columns
                ]
                st.dataframe(sb_unplayed[sb_cols].sort_values("match_date"), use_container_width=True)
            else:
                st.info("No unplayed upcoming fixtures returned for the selected leagues right now.")

        with tab_roi:
            source_choice = st.radio(
                "Settlement & P&L Source",
                options=["Supabase Settled Matches", "Walk-Forward Backtest"],
                horizontal=True,
            )
            if source_choice == "Supabase Settled Matches":
                settled_df, kelly_stats = settle_predictions_with_kelly(sb_settled, min_edge=min_edge)
            else:
                settled_df, kelly_stats = settle_predictions_with_kelly(backtest_df, min_edge=min_edge)

            st.subheader("Fractional Kelly Profit / Loss Curve")
            render_kelly_pnl_chart(settled_df, kelly_stats, height=250)

            st.divider()
            st.write(
                f"**Flat-Unit Win Rate:** {roi_stats['win_rate']:.2%} "
                f"({int(roi_stats['wins'])}/{int(roi_stats['bets'])} bets at edge >= {min_edge_pct:.1f}%)"
            )
            league_rows = []
            for code in selected_leagues:
                sub = backtest_df[backtest_df["league_code"] == code]
                if not sub.empty:
                    l_acc = float((sub["predicted_outcome"] == sub["result"]).mean())
                    l_brier = float(sub["brier_score"].mean())
                    l_roi = compute_simulated_roi(sub, min_edge=min_edge)
                    league_rows.append(
                        {
                            "League": f"{code} — {LEAGUE_NAMES.get(code, code)}",
                            "Matches": len(sub),
                            "Accuracy": round(l_acc, 4),
                            "Brier": round(l_brier, 4),
                            "Bets": int(l_roi["bets"]),
                            "ROI": round(l_roi["roi"], 4),
                            "Units": round(l_roi["profit_units"], 2),
                        }
                    )
            if league_rows:
                st.dataframe(pd.DataFrame(league_rows), use_container_width=True)

        with tab_history:
            hist_cols = [
                "match_date",
                "league_code",
                "home_team",
                "away_team",
                "prob_home",
                "prob_draw",
                "prob_away",
                "predicted_outcome",
                "result",
                "brier_score",
            ]
            st.dataframe(
                backtest_df[hist_cols].sort_values("match_date", ascending=False).head(50),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Engine error: {exc}")


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit_runtime():
    render_streamlit_app()
elif __name__ == "__main__":
    raw = load_english_football_data(leagues=["E0", "E1", "E2", "E3"])
    feat = engineer_prematch_features(raw)
    backtest_df, brier = run_walk_forward_backtest(feat)
    acc = float((backtest_df["predicted_outcome"] == backtest_df["result"]).mean())
    roi_stats = compute_simulated_roi(backtest_df, min_edge=0.03)
    synced = sync_predictions_to_supabase(backtest_df)

    print(f"Loaded {len(raw)} English football matches.")
    print(f"Walk-forward evaluated matches: {len(backtest_df)}")
    print(f"Out-of-sample Accuracy: {acc:.2%} | Multi-class Brier Score: {brier:.4f}")
    print(f"Simulated ROI (>=3% edge): {roi_stats['roi']:+.2%} ({roi_stats['profit_units']:+.2f}u)")
    print(f"Synced rows to Supabase: {synced}")
