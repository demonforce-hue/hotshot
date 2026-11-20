import argparse
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from football_engine import (
    DEFAULT_SEASONS,
    LEAGUE_NAMES,
    build_match_id,
    engineer_prematch_features,
    load_english_football_data,
    run_walk_forward_backtest,
    sync_predictions_to_supabase,
)


def fetch_upcoming_fixtures(leagues: Optional[List[str]] = None, timeout: int = 20) -> pd.DataFrame:
    """
    Attempts to load upcoming English football fixtures from football-data.co.uk/fixtures.csv
    and normalizes them into the same schema used by load_english_football_data().
    """
    target_leagues = set(leagues or ["E0", "E1", "E2", "E3"])
    url = "https://www.football-data.co.uk/fixtures.csv"

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text))
    except Exception:
        return pd.DataFrame()

    if df.empty or not {"HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    if "Div" in df.columns:
        df["league_code"] = df["Div"].astype(str).str.strip()
        df = df[df["league_code"].isin(target_leagues)].copy()
    else:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    parsed_dates = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, "Date"], format="%d/%m/%y", errors="coerce"
        )

    home_team_s = df["HomeTeam"].astype(str).str.strip()
    away_team_s = df["AwayTeam"].astype(str).str.strip()
    match_date_s = parsed_dates.dt.strftime("%Y-%m-%d")

    season_code = DEFAULT_SEASONS[-1] if DEFAULT_SEASONS else "2627"

    numeric_cols: Dict[str, pd.Series] = {}
    for col in ["B365H", "B365D", "B365A"]:
        if col in df.columns:
            numeric_cols[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            numeric_cols[col] = pd.Series(np.nan, index=df.index, dtype="float64")

    match_id_s = pd.Series(
        [
            build_match_id(l, d, h, a)
            for l, d, h, a in zip(df["league_code"], match_date_s, home_team_s, away_team_s)
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
            "home_team": home_team_s.reset_index(drop=True),
            "away_team": away_team_s.reset_index(drop=True),
            "home_goals": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "away_goals": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "result": pd.Series(np.nan, index=range(len(df)), dtype="object"),
            "HS": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "AS": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "HST": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "AST": pd.Series(np.nan, index=range(len(df)), dtype="float64"),
            "B365H": numeric_cols["B365H"].reset_index(drop=True),
            "B365D": numeric_cols["B365D"].reset_index(drop=True),
            "B365A": numeric_cols["B365A"].reset_index(drop=True),
        }
    )

    out = out.dropna(subset=["match_date_dt", "home_team", "away_team"]).copy()
    return out.sort_values(["match_date_dt", "league_code", "match_id"]).reset_index(drop=True)


def compute_simulated_roi(eval_df: pd.DataFrame, min_edge: float = 0.03) -> Dict[str, float]:
    """
    Simulates flat 1-unit bets whenever the model's estimated probability exceeds
    bookmaker implied probability by at least `min_edge` on a completed match.
    """
    df = eval_df[eval_df["result"].isin(["H", "D", "A"])].copy()
    if df.empty:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "profit_units": 0.0, "roi": 0.0}

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
    """
    Trains LightGBM on all completed historical matches in `df_feat` and scores
    unplayed `upcoming_df` rows using their pre-match features.
    """
    if upcoming_df.empty:
        return pd.DataFrame()

    label_map = {"H": 0, "D": 1, "A": 2}
    inv_map = {0: "H", 1: "D", 2: "A"}

    completed = df_feat[df_feat["result"].isin(label_map.keys())].copy()
    if len(completed) < min_train_rows:
        return pd.DataFrame()

    from football_engine import FEATURE_COLS
    import lightgbm as lgb

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

    for side, p_col, odds_col, e_col in [
        ("H", "prob_home", "B365H", "edge_home"),
        ("D", "prob_draw", "B365D", "edge_draw"),
        ("A", "prob_away", "B365A", "edge_away"),
    ]:
        odds = pd.to_numeric(up_feat[odds_col], errors="coerce")
        imp = np.where(odds > 1.0, 1.0 / odds, np.nan)
        up_feat[f"imp_prob_{side.lower()}"] = imp
        up_feat[e_col] = up_feat[p_col] - imp

    return up_feat.copy()


def run_daily(
    leagues: List[str],
    min_edge: float = 0.03,
    sync: bool = True,
    sync_limit: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    raw = load_english_football_data(leagues=leagues)
    upcoming_raw = fetch_upcoming_fixtures(leagues=leagues)

    combined = raw.copy()
    if not upcoming_raw.empty:
        existing_ids = set(combined["match_id"].tolist())
        new_up = upcoming_raw[~upcoming_raw["match_id"].isin(existing_ids)].copy()
        if not new_up.empty:
            combined = pd.concat([combined, new_up], ignore_index=True).copy()

    feat = engineer_prematch_features(combined)
    backtest_df, brier = run_walk_forward_backtest(feat)
    roi_stats = compute_simulated_roi(backtest_df, min_edge=min_edge)

    unplayed_feat = feat[~feat["result"].isin(["H", "D", "A"])].copy()
    upcoming_scored = score_upcoming_with_latest_model(feat, unplayed_feat)

    to_sync = backtest_df.tail(sync_limit).copy()
    if not upcoming_scored.empty:
        to_sync = pd.concat([to_sync, upcoming_scored], ignore_index=True).drop_duplicates(
            subset=["match_id"], keep="last"
        )

    synced_count = 0
    if sync:
        synced_count = sync_predictions_to_supabase(to_sync, limit=len(to_sync))

    acc = float((backtest_df["predicted_outcome"] == backtest_df["result"]).mean())
    print(f"Leagues: {', '.join(f'{c} ({LEAGUE_NAMES.get(c, c)})' for c in leagues)}")
    print(f"Walk-forward matches: {len(backtest_df)} | Accuracy: {acc:.2%} | Brier: {brier:.4f}")
    print(
        f"Edge (>={min_edge:.1%}) sim: {int(roi_stats['bets'])} bets | "
        f"Win rate: {roi_stats['win_rate']:.2%} | "
        f"Profit: {roi_stats['profit_units']:+.2f}u | ROI: {roi_stats['roi']:+.2%}"
    )
    print(f"Upcoming scored fixtures: {len(upcoming_scored)} | Synced to Supabase: {synced_count}")

    return backtest_df, upcoming_scored, roi_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily English Football AI Runner")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["E0", "E1", "E2", "E3"],
        help="League codes to run (default: E0 E1 E2 E3)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum probability edge vs bookmaker implied probability for ROI sim",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Run pipeline without syncing to Supabase",
    )
    args = parser.parse_args()
    run_daily(leagues=args.leagues, min_edge=args.min_edge, sync=not args.no_sync)


if __name__ == "__main__":
    main()
