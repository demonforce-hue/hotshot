import hashlib
import os
from io import StringIO
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

LEAGUE_NAMES: Dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

# Season codes used by football-data.co.uk (e.g. '2425' = 2024/25)
DEFAULT_SEASONS: List[str] = ["2122", "2223", "2324", "2425", "2526", "2627"]


def build_match_id(league_code: str, match_date: str, home_team: str, away_team: str) -> str:
    raw = f"{league_code}|{match_date}|{home_team.strip()}|{away_team.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def fetch_season_csv(season: str, league_code: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or not {"HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
            return None
        df["league_code"] = league_code
        df["season"] = season
        return df
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

    df = pd.concat(frames, ignore_index=True)

    # Parse dates flexibly (DD/MM/YYYY or DD/MM/YY)
    parsed_dates = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, "Date"], format="%d/%m/%y", errors="coerce"
        )

    df["match_date_dt"] = parsed_dates
    df = df.dropna(subset=["match_date_dt", "HomeTeam", "AwayTeam"]).copy()
    df["home_team"] = df["HomeTeam"].astype(str).str.strip()
    df["away_team"] = df["AwayTeam"].astype(str).str.strip()
    df["match_date"] = df["match_date_dt"].dt.strftime("%Y-%m-%d")

    for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["home_goals"] = df["FTHG"]
    df["away_goals"] = df["FTAG"]
    df["result"] = df["FTR"] if "FTR" in df.columns else np.nan
    df.loc[~df["result"].isin(["H", "D", "A"]), "result"] = np.nan

    df["match_id"] = [
        build_match_id(l, d, h, a)
        for l, d, h, a in zip(df["league_code"], df["match_date"], df["home_team"], df["away_team"])
    ]

    df = df.sort_values(["match_date_dt", "league_code", "match_id"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    return df


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

        # Bookmaker implied probabilities (normalized)
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

        # Update Elo & history ONLY if match has a completed result
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
    return pd.concat([data, feat_df], axis=1)


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


def run_walk_forward_backtest(df_feat: pd.DataFrame, min_train_rows: int = 500) -> Tuple[pd.DataFrame, float]:
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

    eval_df["prob_home"] = probs[:, 0]
    eval_df["prob_draw"] = probs[:, 1]
    eval_df["prob_away"] = probs[:, 2]
    eval_df["predicted_outcome"] = [inv_map[int(i)] for i in np.argmax(probs, axis=1)]

    # Multi-class Brier score
    y_one_hot = np.eye(3)[y_all.loc[valid_mask].to_numpy()]
    brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
    eval_df["brier_score"] = np.sum((probs - y_one_hot) ** 2, axis=1)

    # Edges vs implied market probabilities
    eval_df["edge_home"] = eval_df["prob_home"] - eval_df["imp_prob_h"]
    eval_df["edge_draw"] = eval_df["prob_draw"] - eval_df["imp_prob_d"]
    eval_df["edge_away"] = eval_df["prob_away"] - eval_df["imp_prob_a"]

    return eval_df, brier


def sync_predictions_to_supabase(eval_df: pd.DataFrame, limit: int = 250) -> int:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return 0

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

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

    r1 = requests.post(f"{url}/rest/v1/football_matches?on_conflict=match_id", headers=headers, json=match_payload, timeout=20)
    r2 = requests.post(f"{url}/rest/v1/football_predictions?on_conflict=match_id", headers=headers, json=pred_payload, timeout=20)
    r1.raise_for_status()
    r2.raise_for_status()
    return len(pred_payload)


if __name__ == "__main__":
    raw = load_english_football_data(leagues=["E0", "E1", "E2", "E3"])
    feat = engineer_prematch_features(raw)
    backtest_df, brier = run_walk_forward_backtest(feat)
    acc = float((backtest_df["predicted_outcome"] == backtest_df["result"]).mean())
    synced = sync_predictions_to_supabase(backtest_df)

    print(f"Loaded {len(raw)} English football matches.")
    print(f"Walk-forward evaluated matches: {len(backtest_df)}")
    print(f"Out-of-sample Accuracy: {acc:.2%} | Multi-class Brier Score: {brier:.4f}")
    print(f"Synced rows to Supabase: {synced}")
