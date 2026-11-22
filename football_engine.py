import csv
import hashlib
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st

FOOTBALL_LEAGUE_REGISTRY: Dict[str, Dict[str, Dict[str, str]]] = {
    "England": {
        "E0": {"name": "Premier League", "source": "season"},
        "E1": {"name": "Championship", "source": "season"},
        "E2": {"name": "League One", "source": "season"},
        "E3": {"name": "League Two", "source": "season"},
    },
    "Germany": {
        "D1": {"name": "Bundesliga", "source": "season"},
        "D2": {"name": "2. Bundesliga", "source": "season"},
    },
    "Spain": {
        "SP1": {"name": "La Liga", "source": "season"},
        "SP2": {"name": "Segunda División", "source": "season"},
    },
    "Italy": {
        "I1": {"name": "Serie A", "source": "season"},
        "I2": {"name": "Serie B", "source": "season"},
    },
    "France": {
        "F1": {"name": "Ligue 1", "source": "season"},
        "F2": {"name": "Ligue 2", "source": "season"},
    },
    "Global / Extra": {
        "BRA": {"name": "Brazil Série A", "source": "extra"},
        "USA": {"name": "USA MLS", "source": "extra"},
        "JPN": {"name": "Japan J1 League", "source": "extra"},
        "ARG": {"name": "Argentina Primera", "source": "extra"},
    },
}


def get_all_league_meta() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for country, leagues in FOOTBALL_LEAGUE_REGISTRY.items():
        for code, meta in leagues.items():
            out[code] = {
                "name": meta["name"],
                "source": meta["source"],
                "country": country,
            }
    return out


LEAGUE_NAMES: Dict[str, str] = {code: m["name"] for code, m in get_all_league_meta().items()}
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
    "h_rest_days",
    "a_rest_days",
    "rest_diff",
    "h_cong14",
    "a_cong14",
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

_SIDE_TO_IMP = {
    "H": "imp_prob_h",
    "D": "imp_prob_d",
    "A": "imp_prob_a",
}

_SIDE_TO_EDGE = {
    "H": "edge_home",
    "D": "edge_draw",
    "A": "edge_away",
}

_SIDE_TO_ODDS_CANDIDATES = {
    "H": ["best_odds", "B365H", "b365_h"],
    "D": ["best_odds", "B365D", "b365_d"],
    "A": ["best_odds", "B365A", "b365_a"],
}

_SIDE_LABELS = {
    "H": "HOME WIN",
    "D": "DRAW",
    "A": "AWAY WIN",
}

_NEGATIVE_NEWS_TERMS = [
    "ruled out",
    "injured",
    "injury",
    "hamstring",
    "knee",
    "ankle",
    "groin",
    "calf",
    "suspended",
    "suspension",
    "red card",
    "doubtful",
    "surgery",
    "illness",
    "benched",
    "dropped",
    "miss match",
    "out for",
]

_POSITIVE_NEWS_TERMS = [
    "fit to start",
    "returns to training",
    "boost",
    "available",
    "cleared to play",
    "back in squad",
    "full training",
]


def _align_xg_with_1x2(
    lam_h: float,
    lam_a: float,
    prob_h: float,
    prob_d: float,
    prob_a: float,
) -> Tuple[float, float]:
    """
    Gently reconciles total expected goals from the Poisson model with the
    directional home/away balance of the 1X2 classifier so scorelines agree with match calls.
    """
    lh = float(np.clip(lam_h, 0.25, 4.5))
    la = float(np.clip(lam_a, 0.20, 4.5))
    total_xg = float(np.clip(lh + la, 1.10, 5.50))

    raw_share_h = lh / (lh + la)
    # Convert 1X2 probabilities into an implied home goal share
    clf_share_h = float(np.clip(prob_h + 0.5 * prob_d, 0.14, 0.86))

    blended_share_h = (0.35 * raw_share_h) + (0.65 * clf_share_h)
    adj_h = round(float(np.clip(total_xg * blended_share_h, 0.25, 4.5)), 2)
    adj_a = round(float(np.clip(total_xg * (1.0 - blended_share_h), 0.20, 4.5)), 2)
    return adj_h, adj_a


def _poisson_scoreline_summary(lam_h: float, lam_a: float, max_goals: int = 6) -> Tuple[str, float, str]:
    """Compute exact scoreline probabilities from home/away Poisson lambdas."""
    lh = float(np.clip(lam_h, 0.25, 4.5))
    la = float(np.clip(lam_a, 0.20, 4.5))

    p_h = np.array([(lh ** k) * math.exp(-lh) / math.factorial(k) for k in range(max_goals + 1)])
    p_a = np.array([(la ** k) * math.exp(-la) / math.factorial(k) for k in range(max_goals + 1)])
    mat = np.outer(p_h, p_a)

    # Light low-score draw correlation adjustment (Dixon-Coles style)
    mat[0, 0] *= 1.06
    mat[1, 1] *= 1.03
    mat /= mat.sum()

    flat_indices = np.argsort(mat.ravel())[::-1]
    top_items: List[str] = []
    best_score = "1 - 1"
    best_prob = 0.0

    for rank, idx in enumerate(flat_indices[:3]):
        hg = int(idx // (max_goals + 1))
        ag = int(idx % (max_goals + 1))
        prob = float(mat[hg, ag])
        if rank == 0:
            best_score = f"{hg} - {ag}"
            best_prob = prob
        top_items.append(f"{hg}-{ag} ({prob * 100:.1f}%)")

    return best_score, best_prob, " • ".join(top_items)


class BaseNewsPlugin:
    """Standardized news/availability plugin interface for multi-sport expansion."""

    sport: str = "generic"

    def fetch_team_news(
        self,
        teams: List[str],
        match_dates: Optional[Dict[str, str]] = None,
        league_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class FootballNewsPlugin(BaseNewsPlugin):
    """
    Fetches live football availability & headline news signals:
    - FPL bootstrap API for Premier League (E0) player injury/suspension status
    - Google News RSS headline scan for injury/suspension/squad news across leagues
    """

    sport: str = "football"

    def __init__(self, timeout: int = 8, max_rss_teams: int = 12):
        self.timeout = timeout
        self.max_rss_teams = max_rss_teams

    def _fetch_fpl_penalties(self) -> Dict[str, Tuple[float, str]]:
        """Returns {normalized_team_name: (penalty_0_to_1, absence_summary)} from FPL API."""
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if not resp.ok:
                return {}
            data = resp.json()
        except Exception:
            return {}

        teams_raw = data.get("teams", [])
        elements = data.get("elements", [])
        if not teams_raw or not elements:
            return {}

        id_to_team_name: Dict[int, str] = {
            int(t["id"]): str(t.get("name", "")).strip() for t in teams_raw if "id" in t
        }

        team_penalties: Dict[str, float] = {name: 0.0 for name in id_to_team_name.values()}
        team_notes: Dict[str, List[str]] = {name: [] for name in id_to_team_name.values()}

        for p in elements:
            tid = p.get("team")
            if tid not in id_to_team_name:
                continue
            t_name = id_to_team_name[tid]
            status = str(p.get("status", "a")).lower()
            chance = p.get("chance_of_next_round")
            minutes = float(p.get("minutes") or 0.0)
            ict = float(p.get("ict_index") or 0.0)

            importance = float(np.clip((minutes / 1800.0) * 0.7 + (ict / 180.0) * 0.3, 0.08, 1.0))

            sev = 0.0
            if status in ("i", "s", "u"):
                sev = 0.22 * importance
            elif status == "d":
                c_val = float(chance) / 100.0 if chance is not None else 0.5
                sev = (1.0 - c_val) * 0.16 * importance

            if sev > 0.0:
                team_penalties[t_name] = min(1.0, team_penalties[t_name] + sev)
                if importance >= 0.22 and len(team_notes[t_name]) < 3:
                    web_name = str(p.get("web_name", "")).strip()
                    news_txt = str(p.get("news", "")).strip()
                    if web_name:
                        tag = f"{web_name} ({status.upper()})"
                        if news_txt:
                            tag = f"{web_name}: {news_txt[:45]}"
                        team_notes[t_name].append(tag)

        out: Dict[str, Tuple[float, str]] = {}
        for t_name, pen in team_penalties.items():
            notes_str = "; ".join(team_notes.get(t_name, []))
            out[t_name.lower()] = (round(float(pen), 4), notes_str)
        return out

    def _fetch_rss_signal(self, team: str, league_code: str) -> Tuple[float, float, str]:
        """Returns (extra_penalty, sentiment_-1_to_1, headline_snippet) via Google News RSS."""
        lg_label = LEAGUE_NAMES.get(league_code, "football")
        query = quote_plus(f"{team} {lg_label} injury OR squad OR suspended OR lineup")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-GB&gl=GB&ceid=GB:en"
        try:
            resp = requests.get(rss_url, timeout=self.timeout)
            if not resp.ok or not resp.text.strip():
                return 0.0, 0.0, ""
            root = ET.fromstring(resp.text)
            titles = [
                (item.findtext("title") or "").strip()
                for item in root.findall(".//item")[:5]
            ]
        except Exception:
            return 0.0, 0.0, ""

        if not titles:
            return 0.0, 0.0, ""

        joined_low = " ".join(titles).lower()
        neg_hits = sum(1 for term in _NEGATIVE_NEWS_TERMS if term in joined_low)
        pos_hits = sum(1 for term in _POSITIVE_NEWS_TERMS if term in joined_low)

        penalty = float(np.clip(neg_hits * 0.04, 0.0, 0.28))
        raw_sent = pos_hits - neg_hits
        sentiment = float(np.clip(raw_sent / 4.0, -1.0, 1.0))

        top_title = titles[0].split(" - ")[0].strip()
        snippet = f"News: {top_title[:65]}" if (neg_hits > 0 or pos_hits > 0) else ""
        return round(penalty, 4), round(sentiment, 4), snippet

    @staticmethod
    def _match_fpl_penalty(team: str, fpl_map: Dict[str, Tuple[float, str]]) -> Tuple[float, str]:
        if not fpl_map:
            return 0.0, ""
        t_low = team.lower().strip()
        if t_low in fpl_map:
            return fpl_map[t_low]
        aliases = {
            "man united": "manchester utd",
            "manchester united": "manchester utd",
            "man city": "manchester city",
            "spurs": "tottenham",
            "tottenham hotspur": "tottenham",
            "wolves": "wolverhampton",
            "wolverhampton wanderers": "wolverhampton",
            "nott'm forest": "nott'm forest",
            "nottingham forest": "nott'm forest",
            "newcastle united": "newcastle",
            "west ham united": "west ham",
            "brighton & hove albion": "brighton",
            "leicester city": "leicester",
            "ipswich town": "ipswich",
        }
        mapped = aliases.get(t_low)
        if mapped and mapped in fpl_map:
            return fpl_map[mapped]
        for k, val in fpl_map.items():
            if len(k) >= 4 and (k in t_low or t_low in k):
                return val
        return 0.0, ""

    def fetch_team_news(
        self,
        teams: List[str],
        match_dates: Optional[Dict[str, str]] = None,
        league_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        match_dates = match_dates or {}
        league_map = league_map or {}
        unique_teams = [t for t in dict.fromkeys([str(x).strip() for x in teams if str(x).strip()])]

        if not unique_teams:
            return pd.DataFrame(
                columns=[
                    "team",
                    "match_date",
                    "availability_penalty",
                    "news_sentiment",
                    "key_absences",
                    "source_tag",
                ]
            )

        fpl_map = self._fetch_fpl_penalties() if any(league_map.get(t) == "E0" for t in unique_teams) else {}

        rows: List[Dict[str, object]] = []
        for idx, team in enumerate(unique_teams):
            lg = league_map.get(team, "")
            fpl_pen, fpl_note = self._match_fpl_penalty(team, fpl_map)

            rss_pen = 0.0
            rss_sent = 0.0
            rss_note = ""
            if idx < self.max_rss_teams:
                rss_pen, rss_sent, rss_note = self._fetch_rss_signal(team, lg)

            combined_pen = float(np.clip(max(fpl_pen, rss_pen) + 0.25 * min(fpl_pen, rss_pen), 0.0, 1.0))
            if fpl_pen > 0 and rss_sent == 0.0:
                sentiment = float(np.clip(-combined_pen * 1.5, -1.0, 1.0))
            else:
                sentiment = float(np.clip(rss_sent - combined_pen * 0.5, -1.0, 1.0))

            notes_parts = [p for p in [fpl_note, rss_note] if p]
            absences_str = " | ".join(notes_parts) if notes_parts else "No major absences flagged"

            if fpl_pen > 0 and rss_pen > 0:
                src_tag = "FPL+RSS"
            elif fpl_pen > 0:
                src_tag = "FPL"
            elif rss_pen > 0:
                src_tag = "RSS"
            else:
                src_tag = "Baseline"

            rows.append(
                {
                    "team": team,
                    "match_date": match_dates.get(team, ""),
                    "availability_penalty": round(combined_pen, 4),
                    "news_sentiment": round(sentiment, 4),
                    "key_absences": absences_str,
                    "source_tag": src_tag,
                }
            )

        return pd.DataFrame(rows)


def apply_news_plugin_to_upcoming(
    upcoming_df: pd.DataFrame,
    plugin: Optional[BaseNewsPlugin] = None,
) -> pd.DataFrame:
    """
    Applies a sport news plugin to upcoming fixtures, adjusting Elo/probabilities/scorelines
    and attaching human-readable news summaries without altering historical training data.
    """
    if upcoming_df.empty:
        return upcoming_df.copy()

    out = upcoming_df.copy()
    active_plugin = plugin or FootballNewsPlugin()

    team_dates: Dict[str, str] = {}
    team_leagues: Dict[str, str] = {}
    all_teams: List[str] = []

    for _, r in out.iterrows():
        m_date = str(r.get("match_date", ""))
        lg = str(r.get("league_code", ""))
        for col in ("home_team", "away_team"):
            t = str(r.get(col, "")).strip()
            if t:
                all_teams.append(t)
                if t not in team_dates or (m_date and m_date < team_dates[t]):
                    team_dates[t] = m_date
                if t not in team_leagues and lg:
                    team_leagues[t] = lg

    news_df = active_plugin.fetch_team_news(all_teams, match_dates=team_dates, league_map=team_leagues)
    lookup: Dict[str, Dict[str, object]] = {}
    if not news_df.empty:
        for _, nr in news_df.iterrows():
            lookup[str(nr["team"]).strip()] = {
                "penalty": float(pd.to_numeric(nr.get("availability_penalty"), errors="coerce") or 0.0),
                "sentiment": float(pd.to_numeric(nr.get("news_sentiment"), errors="coerce") or 0.0),
                "absences": str(nr.get("key_absences", "No major absences flagged")),
                "source": str(nr.get("source_tag", "Baseline")),
            }

    h_pen_list: List[float] = []
    a_pen_list: List[float] = []
    elo_shift_list: List[float] = []
    news_summary_list: List[str] = []

    adj_p_h: List[float] = []
    adj_p_d: List[float] = []
    adj_p_a: List[float] = []
    adj_best_sides: List[str] = []
    adj_best_edges: List[float] = []
    adj_scores: List[str] = []
    adj_score_probs: List[float] = []
    adj_top_scores: List[str] = []
    adj_xg_h: List[float] = []
    adj_xg_a: List[float] = []

    max_elo_swing = 140.0

    for _, r in out.iterrows():
        ht = str(r.get("home_team", "")).strip()
        at = str(r.get("away_team", "")).strip()

        h_info = lookup.get(ht, {"penalty": 0.0, "sentiment": 0.0, "absences": "Baseline", "source": "Baseline"})
        a_info = lookup.get(at, {"penalty": 0.0, "sentiment": 0.0, "absences": "Baseline", "source": "Baseline"})

        hp = float(h_info["penalty"])
        ap = float(a_info["penalty"])
        hs = float(h_info["sentiment"])
        as_ = float(a_info["sentiment"])

        h_net = (-hp * 0.85) + (hs * 0.15)
        a_net = (-ap * 0.85) + (as_ * 0.15)
        net_diff = float(np.clip(h_net - a_net, -1.0, 1.0))
        elo_shift = round(net_diff * max_elo_swing, 1)

        h_pen_list.append(round(hp, 4))
        a_pen_list.append(round(ap, 4))
        elo_shift_list.append(elo_shift)
        news_summary_list.append(
            f"{ht} [{h_info['source']}]: {h_info['absences']} // {at} [{a_info['source']}]: {a_info['absences']}"
        )

        base_h = float(pd.to_numeric(r.get("prob_home"), errors="coerce") or 0.34)
        base_d = float(pd.to_numeric(r.get("prob_draw"), errors="coerce") or 0.28)
        base_a = float(pd.to_numeric(r.get("prob_away"), errors="coerce") or 0.38)

        prob_swing = float(np.clip(elo_shift / 800.0, -0.14, 0.14))
        nh = float(np.clip(base_h + prob_swing, 0.03, 0.92))
        na = float(np.clip(base_a - prob_swing, 0.03, 0.92))
        nd = float(np.clip(1.0 - nh - na, 0.05, 0.55))
        tot_p = nh + nd + na
        nh, nd, na = nh / tot_p, nd / tot_p, na / tot_p

        adj_p_h.append(nh)
        adj_p_d.append(nd)
        adj_p_a.append(na)

        side_map = [
            ("H", nh, r.get("imp_prob_h"), r.get("B365H")),
            ("D", nd, r.get("imp_prob_d"), r.get("B365D")),
            ("A", na, r.get("imp_prob_a"), r.get("B365A")),
        ]
        valid_sides = []
        for s_code, p_val, imp_v, odd_v in side_map:
            odd_num = pd.to_numeric(odd_v, errors="coerce")
            imp_num = pd.to_numeric(imp_v, errors="coerce")
            if pd.isna(imp_num) and pd.notna(odd_num) and float(odd_num) > 1.0:
                imp_num = 1.0 / float(odd_num)
            if pd.notna(imp_num) and pd.notna(odd_num) and float(odd_num) > 1.0:
                valid_sides.append((s_code, float(p_val - float(imp_num))))

        if valid_sides:
            top_s, top_e = max(valid_sides, key=lambda item: item[1])
            adj_best_sides.append(top_s)
            adj_best_edges.append(round(top_e, 4))
        else:
            top_idx = int(np.argmax([nh, nd, na]))
            adj_best_sides.append(["H", "D", "A"][top_idx])
            adj_best_edges.append(np.nan)

        ex_h = pd.to_numeric(r.get("exp_goals_h"), errors="coerce")
        ex_a = pd.to_numeric(r.get("exp_goals_a"), errors="coerce")
        if pd.notna(ex_h) and pd.notna(ex_a):
            xg_mult_h = float(np.clip(1.0 + net_diff * 0.22, 0.65, 1.35))
            xg_mult_a = float(np.clip(1.0 - net_diff * 0.22, 0.65, 1.35))
            raw_xh = float(np.clip(float(ex_h) * xg_mult_h, 0.25, 4.5))
            raw_xa = float(np.clip(float(ex_a) * xg_mult_a, 0.20, 4.5))
            n_xh, n_xa = _align_xg_with_1x2(raw_xh, raw_xa, nh, nd, na)
            adj_xg_h.append(n_xh)
            adj_xg_a.append(n_xa)
            sc_str, sc_pr, top_str = _poisson_scoreline_summary(n_xh, n_xa)
            adj_scores.append(sc_str)
            adj_score_probs.append(sc_pr)
            adj_top_scores.append(top_str)
        else:
            adj_xg_h.append(np.nan)
            adj_xg_a.append(np.nan)
            adj_scores.append(str(r.get("pred_score") or "N/A"))
            adj_score_probs.append(float(pd.to_numeric(r.get("pred_score_prob"), errors="coerce") or 0.0))
            adj_top_scores.append(str(r.get("top_scores") or ""))

    out["h_news_penalty"] = h_pen_list
    out["a_news_penalty"] = a_pen_list
    out["news_elo_shift"] = elo_shift_list
    out["news_summary"] = news_summary_list

    out["raw_prob_home"] = out["prob_home"]
    out["raw_prob_draw"] = out["prob_draw"]
    out["raw_prob_away"] = out["prob_away"]
    out["prob_home"] = adj_p_h
    out["prob_draw"] = adj_p_d
    out["prob_away"] = adj_p_a

    out["edge_home"] = out["prob_home"] - pd.to_numeric(out["imp_prob_h"], errors="coerce")
    out["edge_draw"] = out["prob_draw"] - pd.to_numeric(out["imp_prob_d"], errors="coerce")
    out["edge_away"] = out["prob_away"] - pd.to_numeric(out["imp_prob_a"], errors="coerce")

    out["best_value_side"] = adj_best_sides
    out["best_edge"] = adj_best_edges

    if "exp_goals_h" in out.columns:
        out["exp_goals_h"] = adj_xg_h
        out["exp_goals_a"] = adj_xg_a
        out["pred_score"] = adj_scores
        out["pred_score_prob"] = adj_score_probs
        out["top_scores"] = adj_top_scores

    return out


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
    """Joins wrapped continuation lines and drops blank lines in CSV feeds."""
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


def _normalize_league_df(df: pd.DataFrame, league_code: str, season_label: str) -> Optional[pd.DataFrame]:
    """Normalizes both per-season (/mmz4281/) and global (/new/) football-data CSVs."""
    if df.empty:
        return None

    home_col = next((c for c in ["HomeTeam", "Home"] if c in df.columns), None)
    away_col = next((c for c in ["AwayTeam", "Away"] if c in df.columns), None)
    date_col = next((c for c in ["Date"] if c in df.columns), None)
    if not home_col or not away_col or not date_col:
        return None

    hg_col = next((c for c in ["FTHG", "HG"] if c in df.columns), None)
    ag_col = next((c for c in ["FTAG", "AG"] if c in df.columns), None)
    res_col = next((c for c in ["FTR", "Res"] if c in df.columns), None)

    parsed_dates = pd.to_datetime(df[date_col].astype(str).str.strip(), format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed_dates.isna()
    if missing_mask.any():
        parsed_dates.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, date_col].astype(str).str.strip(), format="%d/%m/%y", errors="coerce"
        )
    iso_mask = parsed_dates.isna()
    if iso_mask.any():
        parsed_dates.loc[iso_mask] = pd.to_datetime(
            df.loc[iso_mask, date_col].astype(str).str.strip(), errors="coerce"
        )

    def _num(col_name: Optional[str]) -> pd.Series:
        if col_name and col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    home_team_s = df[home_col].astype(str).str.strip()
    away_team_s = df[away_col].astype(str).str.strip()
    match_date_s = parsed_dates.dt.strftime("%Y-%m-%d")

    if res_col and res_col in df.columns:
        result_s = df[res_col].astype(str).str.strip().str.upper()
        result_s = result_s.where(result_s.isin(["H", "D", "A"]), np.nan)
    else:
        result_s = pd.Series(np.nan, index=df.index, dtype="object")

    match_id_s = pd.Series(
        [
            build_match_id(league_code, d, h, a)
            for d, h, a in zip(match_date_s, home_team_s, away_team_s)
        ],
        index=df.index,
        dtype="object",
    )

    out = pd.DataFrame(
        {
            "league_code": pd.Series(league_code, index=df.index, dtype="object"),
            "season": pd.Series(season_label, index=df.index, dtype="object"),
            "match_id": match_id_s,
            "match_date_dt": parsed_dates,
            "match_date": match_date_s,
            "home_team": home_team_s,
            "away_team": away_team_s,
            "home_goals": _num(hg_col),
            "away_goals": _num(ag_col),
            "result": result_s,
            "HS": _num("HS"),
            "AS": _num("AS"),
            "HST": _num("HST"),
            "AST": _num("AST"),
            "B365H": _num("B365H"),
            "B365D": _num("B365D"),
            "B365A": _num("B365A"),
        }
    )
    out = out.dropna(subset=["match_date_dt", "home_team", "away_team"]).copy()
    return out


def fetch_season_csv(season: str, league_code: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = _read_csv_resilient(resp.text, header=0)
        return _normalize_league_df(df, league_code=league_code, season_label=season)
    except Exception:
        return None


def fetch_extra_league_csv(league_code: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    url = f"https://www.football-data.co.uk/new/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = _read_csv_resilient(resp.text, header=0)
        return _normalize_league_df(df, league_code=league_code, season_label="GLOBAL")
    except Exception:
        return None


def load_english_football_data(
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    leagues = leagues or ["E0", "E1"]
    seasons = seasons or DEFAULT_SEASONS
    meta_map = get_all_league_meta()

    frames: List[pd.DataFrame] = []
    for league in leagues:
        src_type = meta_map.get(league, {}).get("source", "season")
        if src_type == "extra":
            extra_df = fetch_extra_league_csv(league)
            if extra_df is not None and not extra_df.empty:
                frames.append(extra_df)
        else:
            for season in seasons:
                raw_df = fetch_season_csv(season, league)
                if raw_df is not None and not raw_df.empty:
                    frames.append(raw_df)

    if not frames:
        raise RuntimeError("No football data could be loaded from football-data.co.uk for the selected leagues.")

    clean_df = pd.concat(frames, ignore_index=True).copy()
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


def _resolve_value_bet_side(row: pd.Series, min_edge: float = 0.03) -> Optional[str]:
    """Pick the side with highest positive edge >= min_edge, or fall back to best_value_side."""
    edge_scores: Dict[str, float] = {}
    for side, e_col in _SIDE_TO_EDGE.items():
        val = pd.to_numeric(row.get(e_col), errors="coerce")
        if pd.notna(val) and float(val) >= float(min_edge):
            edge_scores[side] = float(val)

    if edge_scores:
        return max(edge_scores.items(), key=lambda kv: kv[1])[0]

    best_edge = pd.to_numeric(row.get("best_edge"), errors="coerce")
    best_side = str(row.get("best_value_side", "")).strip().upper()
    if best_side in ("H", "D", "A") and (pd.isna(best_edge) or float(best_edge) >= float(min_edge)):
        return best_side

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
    min_edge: float = 0.03,
) -> pd.DataFrame:
    """Attach kelly_side, kelly_prob, kelly_odds, kelly_edge, and kelly_stake_pct."""
    out = df.copy()
    if out.empty:
        for col in ["kelly_side", "kelly_prob", "kelly_odds", "kelly_edge", "kelly_stake_pct"]:
            out[col] = pd.Series(dtype="float64")
        return out

    sides: List[Optional[str]] = []
    probs: List[float] = []
    odds_list: List[float] = []
    edges: List[float] = []

    for _, row in out.iterrows():
        side = _resolve_value_bet_side(row, min_edge=min_edge)
        sides.append(side)

        p_col = _SIDE_TO_PROB.get(side or "", "")
        p_val = pd.to_numeric(row.get(p_col), errors="coerce") if p_col else np.nan
        probs.append(float(p_val) if pd.notna(p_val) else np.nan)

        o_val = _resolve_odds_for_side(row, side)
        odds_list.append(o_val)

        e_col = _SIDE_TO_EDGE.get(side or "", "")
        e_val = pd.to_numeric(row.get(e_col), errors="coerce") if e_col else np.nan
        if pd.isna(e_val) and side and side == str(row.get("best_value_side", "")).strip().upper():
            e_val = pd.to_numeric(row.get("best_edge"), errors="coerce")
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

    out["kelly_side"] = sides
    out["kelly_prob"] = np.round(p, 4)
    out["kelly_odds"] = np.round(odds, 3)
    out["kelly_edge"] = np.round(edge_s, 4)
    out["kelly_stake_pct"] = np.round(stake_pct, 4)
    return out


def settle_predictions_with_kelly(
    df: pd.DataFrame,
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.04,
    min_edge: float = 0.03,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Settle rows with compounding fractional Kelly stakes and a 0.0 bankroll floor."""
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
        min_edge=min_edge,
    )

    result_series = work.get("result", pd.Series(index=work.index, dtype="object"))
    actual_series = work.get("actual_outcome", pd.Series(index=work.index, dtype="object"))
    outcome = result_series.combine_first(actual_series).astype(str).str.strip().str.upper()
    work["actual_outcome"] = outcome.where(outcome.isin(["H", "D", "A"]), np.nan)

    sort_cols = [c for c in ["match_date", "match_id"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)

    bankroll = float(starting_bankroll)
    stakes_list: List[float] = []
    pnl_list: List[float] = []
    cum_pnl_list: List[float] = []
    bankroll_list: List[float] = []
    settled_flags: List[bool] = []
    win_flags: List[Optional[bool]] = []

    for _, row in work.iterrows():
        actual = row.get("actual_outcome")
        side = row.get("kelly_side")
        pct = float(pd.to_numeric(row.get("kelly_stake_pct"), errors="coerce") or 0.0)
        odds = float(pd.to_numeric(row.get("kelly_odds"), errors="coerce") or np.nan)

        if actual in ("H", "D", "A") and side in ("H", "D", "A") and pct > 0.0 and pd.notna(odds) and odds > 1.0 and bankroll > 0.0:
            stake = round(bankroll * pct, 2)
            won = side == actual
            trade_pnl = round(stake * (odds - 1.0), 2) if won else round(-stake, 2)
            bankroll = max(0.0, round(bankroll + trade_pnl, 2))
            stakes_list.append(stake)
            pnl_list.append(trade_pnl)
            settled_flags.append(True)
            win_flags.append(bool(won))
        else:
            stakes_list.append(0.0)
            pnl_list.append(np.nan)
            settled_flags.append(False)
            win_flags.append(None)

        cum_pnl_list.append(round(bankroll - float(starting_bankroll), 2))
        bankroll_list.append(bankroll)

    work["stake_amount"] = stakes_list
    work["is_settled_bet"] = settled_flags
    work["is_win"] = win_flags
    work["pnl"] = pnl_list
    work["cum_pnl"] = cum_pnl_list
    work["bankroll_after"] = bankroll_list

    settled_only = work[work["is_settled_bet"]]
    bets = float(len(settled_only))
    wins = float(settled_only["is_win"].fillna(False).sum()) if bets > 0 else 0.0
    total_staked = float(settled_only["stake_amount"].sum()) if bets > 0 else 0.0
    total_pnl = round(bankroll - float(starting_bankroll), 2)
    win_rate = (wins / bets) if bets > 0 else 0.0
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

    stats = {
        "bets": bets,
        "wins": wins,
        "win_rate": float(win_rate),
        "total_staked": round(total_staked, 2),
        "pnl": total_pnl,
        "roi": float(roi),
        "final_bankroll": round(bankroll, 2),
    }
    return work, stats


def render_kelly_pnl_chart(settled_df: pd.DataFrame, stats: Dict[str, float], height: int = 230, compact: bool = False) -> None:
    """Render Kelly ROI KPIs and cumulative Profit/Loss chart in Streamlit."""
    if compact:
        c1, c2 = st.columns(2)
        c1.metric("Kelly ROI", f"{stats['roi']:+.2%}")
        c2.metric("Net P&L", f"{stats['pnl']:+.2f}")
        c3, c4 = st.columns(2)
        c3.metric("Win Rate", f"{stats['win_rate']:.1%}")
        c4.metric("Bankroll", f"{stats['final_bankroll']:.2f}")
    else:
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


def predict_scorelines(df_feat: pd.DataFrame, target_df: pd.DataFrame, min_train_rows: int = 250) -> pd.DataFrame:
    """Fit Poisson LightGBM goal models on completed matches and attach aligned scorelines."""
    if target_df.empty:
        return target_df.copy()

    out = target_df.copy()
    completed = df_feat[
        df_feat["result"].isin(["H", "D", "A"])
        & df_feat["home_goals"].notna()
        & df_feat["away_goals"].notna()
    ].copy()

    if len(completed) < min_train_rows:
        out["exp_goals_h"] = np.nan
        out["exp_goals_a"] = np.nan
        out["pred_score"] = "N/A"
        out["pred_score_prob"] = np.nan
        out["top_scores"] = ""
        return out

    X_train = completed[FEATURE_COLS]
    y_h = pd.to_numeric(completed["home_goals"], errors="coerce").clip(0, 10)
    y_a = pd.to_numeric(completed["away_goals"], errors="coerce").clip(0, 10)

    params = dict(
        objective="poisson",
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=23,
        subsample_for_bin=2000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    model_h = lgb.LGBMRegressor(**params)
    model_a = lgb.LGBMRegressor(**params)
    model_h.fit(X_train, y_h)
    model_a.fit(X_train, y_a)

    lam_h_raw = np.clip(model_h.predict(out[FEATURE_COLS]), 0.25, 4.5)
    lam_a_raw = np.clip(model_a.predict(out[FEATURE_COLS]), 0.20, 4.5)

    p_h_arr = pd.to_numeric(out.get("prob_home"), errors="coerce").fillna(0.42).to_numpy()
    p_d_arr = pd.to_numeric(out.get("prob_draw"), errors="coerce").fillna(0.28).to_numpy()
    p_a_arr = pd.to_numeric(out.get("prob_away"), errors="coerce").fillna(0.30).to_numpy()

    adj_h_list: List[float] = []
    adj_a_list: List[float] = []
    scores: List[str] = []
    score_probs: List[float] = []
    top_lists: List[str] = []

    for lh, la, ph, pd_val, pa in zip(lam_h_raw, lam_a_raw, p_h_arr, p_d_arr, p_a_arr):
        ah, aa = _align_xg_with_1x2(float(lh), float(la), float(ph), float(pd_val), float(pa))
        adj_h_list.append(ah)
        adj_a_list.append(aa)
        sc, pr, top_str = _poisson_scoreline_summary(ah, aa)
        scores.append(sc)
        score_probs.append(pr)
        top_lists.append(top_str)

    out["exp_goals_h"] = adj_h_list
    out["exp_goals_a"] = adj_a_list
    out["pred_score"] = scores
    out["pred_score_prob"] = score_probs
    out["top_scores"] = top_lists
    return out


def _inject_sporty_theme() -> None:
    st.markdown(
        """
        <style>
        .hs-hero-card {
            background: linear-gradient(135deg, #0b1c26 0%, #0f2e2a 60%, #10382d 100%);
            border: 1px solid rgba(45, 212, 191, 0.28);
            border-radius: 16px;
            padding: 18px 20px;
            color: #f0fdf4;
            margin-bottom: 12px;
            max-width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }
        .hs-league-pill {
            display: inline-block;
            background: rgba(20, 184, 166, 0.18);
            border: 1px solid rgba(45, 212, 191, 0.45);
            color: #5eead4;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 3px 10px;
            border-radius: 999px;
            text-transform: uppercase;
            max-width: 100%;
        }
        .hs-matchup-row {
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 8px;
            margin: 14px 0;
            min-width: 0;
        }
        .hs-team-box {
            flex: 1 1 0;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 8px;
            box-sizing: border-box;
        }
        .hs-team-name {
            font-size: clamp(14px, 2.1vw, 19px);
            font-weight: 800;
            color: #ffffff;
            line-height: 1.2;
            min-width: 0;
            overflow-wrap: break-word;
            word-break: normal;
        }
        .hs-team-sub {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 4px;
            line-height: 1.25;
            overflow-wrap: break-word;
        }
        .hs-center-box {
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            min-width: 96px;
            max-width: 100%;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(45, 212, 191, 0.38);
            border-radius: 12px;
            padding: 8px 10px;
            box-sizing: border-box;
        }
        .hs-score-label {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #5eead4;
            text-transform: uppercase;
        }
        .hs-score-val {
            font-size: 22px;
            font-weight: 900;
            color: #ffffff;
            line-height: 1.15;
            margin: 2px 0;
        }
        .hs-score-sub {
            font-size: 11px;
            color: #94a3b8;
        }
        .hs-pick-banner {
            background: rgba(16, 185, 129, 0.14);
            border-left: 4px solid #10b981;
            border-radius: 10px;
            padding: 10px 14px;
            margin-top: 8px;
            min-width: 0;
            overflow-wrap: break-word;
        }
        .hs-pick-title {
            font-size: 15px;
            font-weight: 800;
            color: #6ee7b7;
            overflow-wrap: break-word;
        }
        .hs-pick-meta {
            font-size: 12px;
            color: #cbd5e1;
            margin-top: 2px;
            overflow-wrap: break-word;
        }
        .hs-news-box {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 10px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 11.5px;
            color: #cbd5e1;
            overflow-wrap: break-word;
        }
        .hs-gauge-wrap {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
        }
        .hs-gauge-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 6px;
            font-size: 11.5px;
            color: #cbd5e1;
            margin-bottom: 6px;
        }
        .hs-gauge-track {
            width: 100%;
            height: 8px;
            background: rgba(15, 23, 42, 0.9);
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .hs-gauge-fill {
            height: 100%;
            background: linear-gradient(90deg, #0284c7 0%, #38bdf8 55%, #2dd4bf 100%);
            border-radius: 999px;
        }
        .hs-rail-card {
            background: #0d1721;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 14px 16px;
            max-width: 100%;
            box-sizing: border-box;
        }
        .hs-rail-title {
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #5eead4;
            margin-bottom: 8px;
        }
        .hs-legal-footer {
            background: #0b141d;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 24px;
            color: #cbd5e1;
            font-size: 12px;
            line-height: 1.45;
            max-width: 100%;
            box-sizing: border-box;
            overflow-wrap: break-word;
        }
        .hs-legal-copy {
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 6px;
        }
        .hs-legal-rg {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
            color: #99f6e4;
            font-weight: 600;
        }
        .hs-legal-rg a {
            color: #5eead4;
            text-decoration: underline;
        }
        @media (max-width: 640px) {
            .hs-hero-card {
                padding: 12px 10px;
            }
            .hs-matchup-row {
                gap: 5px;
                margin: 10px 0;
            }
            .hs-team-box {
                padding: 8px 4px;
            }
            .hs-team-name {
                font-size: 11.5px;
                letter-spacing: -0.02em;
            }
            .hs-team-sub {
                font-size: 9.5px;
            }
            .hs-center-box {
                min-width: 72px;
                padding: 6px 5px;
            }
            .hs-score-val {
                font-size: 17px;
            }
            .hs-score-sub {
                font-size: 9px;
            }
            .hs-gauge-meta {
                font-size: 10.5px;
            }
            .hs-legal-footer {
                padding: 12px 12px;
                font-size: 11px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_footer() -> None:
    """Render persistent copyright, speculation disclaimer, and responsible gambling notice."""
    st.markdown(
        """
        <div class="hs-legal-footer">
            <div class="hs-legal-copy">&copy; 2026 Hotshots AI Sports Engine. All rights reserved.</div>
            <div>
                <b>Disclaimer:</b> All information, probabilities, scoreline projections, edge calculations,
                and historical backtest or simulated profit/loss figures provided by Hotshots AI Sports Engine
                remain speculative in nature and are not guaranteed. If any end user chooses to make use of
                this information to place any bets, they do so strictly of their own volition and at their own
                risk, with no guarantees whatsoever, implied or otherwise, from the Hotshots App.
            </div>
            <div class="hs-legal-rg">
                Gamble Aware &bull; Gamble Responsibly &mdash; 18+ only. Please bet within your means.
                Free, confidential support is available at
                <a href="https://www.gambleaware.org/" target="_blank" rel="noopener noreferrer">GambleAware</a>
                and
                <a href="https://www.gamcare.org.uk/" target="_blank" rel="noopener noreferrer">GamCare</a>
                (UK 24/7 Helpline: 0808 8020 133).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sporty_match_deck(
    fixtures_df: pd.DataFrame,
    settled_df: pd.DataFrame,
    kelly_stats: Dict[str, float],
    model_acc: float,
    min_edge: float,
) -> None:
    """Render match-card center + right-rail performance tracker."""
    _inject_sporty_theme()

    if fixtures_df.empty:
        st.info("No upcoming fixtures available to render in the match card deck.")
        return

    options_df = fixtures_df.reset_index(drop=True)
    labels = [
        f"{r['match_date']} • {r['league_code']} • {r['home_team']} vs {r['away_team']} "
        f"[{r.get('pred_score', 'N/A')}] ({r['best_edge']*100:+.1f}%)" if pd.notna(r.get("best_edge")) else
        f"{r['match_date']} • {r['league_code']} • {r['home_team']} vs {r['away_team']} [{r.get('pred_score', 'N/A')}]"
        for _, r in options_df.iterrows()
    ]

    selected_idx = st.selectbox(
        "Select Match Card",
        options=list(range(len(options_df))),
        format_func=lambda idx: labels[idx],
    )
    row = options_df.iloc[int(selected_idx)]

    left_col, right_col = st.columns([2, 1], gap="medium")

    with left_col:
        side = str(row.get("best_value_side") or row.get("predicted_outcome") or "H").strip().upper()
        side_label = _SIDE_LABELS.get(side, side)
        prob_col = _SIDE_TO_PROB.get(side, "prob_home")
        win_prob = float(pd.to_numeric(row.get(prob_col), errors="coerce") or 0.0)
        best_odds = float(pd.to_numeric(row.get("best_odds"), errors="coerce") or 0.0)
        edge_val = float(pd.to_numeric(row.get("best_edge"), errors="coerce") or 0.0)

        h_elo = pd.to_numeric(row.get("h_elo"), errors="coerce")
        a_elo = pd.to_numeric(row.get("a_elo"), errors="coerce")
        h_pts5 = pd.to_numeric(row.get("h_pts5"), errors="coerce")
        a_pts5 = pd.to_numeric(row.get("a_pts5"), errors="coerce")

        pred_score = str(row.get("pred_score") or "N/A")
        score_prob = float(pd.to_numeric(row.get("pred_score_prob"), errors="coerce") or 0.0)
        exp_h = pd.to_numeric(row.get("exp_goals_h"), errors="coerce")
        exp_a = pd.to_numeric(row.get("exp_goals_a"), errors="coerce")
        top_scores = str(row.get("top_scores") or "")

        league_code = str(row.get("league_code", ""))
        league_full = LEAGUE_NAMES.get(league_code, league_code)
        match_date_str = str(row.get("match_date", ""))

        odds_str = f"{best_odds:.2f}" if best_odds > 1.0 else "N/A"
        edge_str = f"{edge_val*100:+.1f}%" if pd.notna(row.get("best_edge")) else "N/A"
        xg_str = f"xG {exp_h:.2f}–{exp_a:.2f}" if pd.notna(exp_h) and pd.notna(exp_a) else ""

        news_shift = pd.to_numeric(row.get("news_elo_shift"), errors="coerce")
        shift_str = f"{news_shift:+.1f} Elo" if pd.notna(news_shift) and float(news_shift) != 0.0 else "0.0 Elo"
        news_summary = str(row.get("news_summary") or "Baseline schedule/news check active")
        prob_pct = float(np.clip(win_prob * 100.0, 0.0, 100.0))

        st.markdown(
            f"""
            <div class="hs-hero-card">
                <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px;">
                    <span class="hs-league-pill">{league_code} • {league_full}</span>
                    <span style="font-size:12px; color:#cbd5e1; font-weight:600;">Kickoff: {match_date_str}</span>
                </div>
                <div class="hs-matchup-row">
                    <div class="hs-team-box">
                        <div class="hs-team-name">{row['home_team']}</div>
                        <div class="hs-team-sub">Elo {h_elo:.0f} • 5G {h_pts5:.2f}</div>
                    </div>
                    <div class="hs-center-box">
                        <div class="hs-score-label">Proj. Score</div>
                        <div class="hs-score-val">{pred_score}</div>
                        <div class="hs-score-sub">{score_prob*100:.1f}% • {xg_str}</div>
                    </div>
                    <div class="hs-team-box">
                        <div class="hs-team-name">{row['away_team']}</div>
                        <div class="hs-team-sub">Elo {a_elo:.0f} • 5G {a_pts5:.2f}</div>
                    </div>
                </div>
                <div class="hs-pick-banner">
                    <div class="hs-pick-title">HOTSHOTS VALUE CALL: {side_label} ({prob_pct:.1f}% Model Prob)</div>
                    <div class="hs-pick-meta">Best Odds: <b>{odds_str}</b> • Model Edge: <b>{edge_str}</b> • News Shift: <b>{shift_str}</b></div>
                </div>
                <div class="hs-news-box">
                    <b>Team News Plugin:</b> {news_summary}
                </div>
                <div class="hs-gauge-wrap">
                    <div class="hs-gauge-meta">
                        <span><b>Model Confidence:</b> {prob_pct:.1f}% (Edge {edge_str})</span>
                        <span>Top: {top_scores}</span>
                    </div>
                    <div class="hs-gauge-track">
                        <div class="hs-gauge-fill" style="width: {prob_pct:.1f}%;"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p_h = float(pd.to_numeric(row.get("prob_home"), errors="coerce") or 0.0)
        p_d = float(pd.to_numeric(row.get("prob_draw"), errors="coerce") or 0.0)
        p_a = float(pd.to_numeric(row.get("prob_away"), errors="coerce") or 0.0)

        i_h = pd.to_numeric(row.get("imp_prob_h"), errors="coerce")
        i_d = pd.to_numeric(row.get("imp_prob_d"), errors="coerce")
        i_a = pd.to_numeric(row.get("imp_prob_a"), errors="coerce")

        f1, f2, f3 = st.columns(3)
        f1.metric("Home (H)", f"{p_h*100:.1f}%", delta=f"Imp {i_h*100:.1f}%" if pd.notna(i_h) else None)
        f2.metric("Draw (D)", f"{p_d*100:.1f}%", delta=f"Imp {i_d*100:.1f}%" if pd.notna(i_d) else None)
        f3.metric("Away (A)", f"{p_a*100:.1f}%", delta=f"Imp {i_a*100:.1f}%" if pd.notna(i_a) else None)

        gd_h = pd.to_numeric(row.get("h_gd5"), errors="coerce")
        gd_a = pd.to_numeric(row.get("a_gd5"), errors="coerce")
        rest_h = pd.to_numeric(row.get("h_rest_days"), errors="coerce")
        rest_a = pd.to_numeric(row.get("a_rest_days"), errors="coerce")

        st.caption(
            f"Key Factors — 5G GD: {row['home_team']} {gd_h:+.2f} vs {row['away_team']} {gd_a:+.2f} "
            f"| Rest: {rest_h:.0f}d vs {rest_a:.0f}d"
        )

    with right_col:
        st.markdown('<div class="hs-rail-card"><div class="hs-rail-title">Model Performance Tracker</div>', unsafe_allow_html=True)
        st.metric("Walk-Forward Accuracy", f"{model_acc:.2%}")
        render_kelly_pnl_chart(settled_df, kelly_stats, height=190, compact=True)
        st.markdown("</div>", unsafe_allow_html=True)


def engineer_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly pre-match features (Elo, rolling form, rest/congestion,
    goal/shot differentials, and market-implied probabilities) with zero leakage.
    """
    data = df.sort_values(["match_date_dt", "match_id"]).reset_index(drop=True).copy()

    elo: Dict[str, float] = {}
    history: Dict[str, List[Dict[str, object]]] = {}

    base_elo = 1500.0
    k_factor = 22.0
    home_advantage_elo = 65.0

    feat_rows: List[Dict[str, float]] = []

    def get_team_stats(team: str, cur_dt: datetime, n: int = 5) -> Dict[str, float]:
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
                "rest_days": 7.0,
                "cong14": 1.0,
            }
        recent = recs[-n:]
        last_dt = recent[-1]["dt"]
        rest_days = float(np.clip((cur_dt - last_dt).total_seconds() / 86400.0, 1.0, 21.0)) if isinstance(last_dt, datetime) else 7.0
        cong14 = float(
            sum(
                1
                for r in recs[-6:]
                if isinstance(r.get("dt"), datetime) and 0.0 <= ((cur_dt - r["dt"]).total_seconds() / 86400.0) <= 14.0
            )
        )
        return {
            "pts_avg": float(np.mean([r["pts"] for r in recent])),
            "gf_avg": float(np.mean([r["gf"] for r in recent])),
            "ga_avg": float(np.mean([r["ga"] for r in recent])),
            "gd_avg": float(np.mean([r["gf"] - r["ga"] for r in recent])),
            "sh_diff_avg": float(np.mean([r["sh_diff"] for r in recent])),
            "sot_diff_avg": float(np.mean([r["sot_diff"] for r in recent])),
            "games_played": float(len(recs)),
            "rest_days": rest_days,
            "cong14": cong14,
        }

    for _, row in data.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        cur_dt = row["match_date_dt"].to_pydatetime() if pd.notna(row.get("match_date_dt")) else datetime(2026, 1, 1)

        h_elo = elo.get(h_team, base_elo)
        a_elo = elo.get(a_team, base_elo)

        h5 = get_team_stats(h_team, cur_dt, n=5)
        a5 = get_team_stats(a_team, cur_dt, n=5)
        h10 = get_team_stats(h_team, cur_dt, n=10)
        a10 = get_team_stats(a_team, cur_dt, n=10)

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
                "h_rest_days": h5["rest_days"],
                "a_rest_days": a5["rest_days"],
                "rest_diff": h5["rest_days"] - a5["rest_days"],
                "h_cong14": h5["cong14"],
                "a_cong14": a5["cong14"],
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
                {
                    "dt": cur_dt,
                    "pts": pts_h,
                    "gf": float(hg),
                    "ga": float(ag),
                    "sh_diff": sh_diff,
                    "sot_diff": sot_diff,
                }
            )
            history.setdefault(a_team, []).append(
                {
                    "dt": cur_dt,
                    "pts": pts_a,
                    "gf": float(ag),
                    "ga": float(hg),
                    "sh_diff": -sh_diff,
                    "sot_diff": -sot_diff,
                }
            )

    feat_df = pd.DataFrame(feat_rows, index=data.index)
    return pd.concat([data, feat_df], axis=1).copy()


def run_walk_forward_backtest(
    df_feat: pd.DataFrame,
    min_train_rows: int = 250,
) -> Tuple[pd.DataFrame, float]:
    label_map = {"H": 0, "D": 1, "A": 2}
    inv_map = {0: "H", 1: "D", 2: "A"}

    completed = df_feat[df_feat["result"].isin(label_map.keys())].copy().reset_index(drop=True)
    if len(completed) <= min_train_rows:
        raise ValueError(f"Not enough completed matches ({len(completed)}) for walk-forward backtest.")

    y_all = completed["result"].map(label_map).astype(int)
    X_all = completed[FEATURE_COLS]

    oof_probs = np.full((len(completed), 3), np.nan)
    step = max(100, min(200, (len(completed) - min_train_rows) // 4 or 100))

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
    min_train_rows: int = 250,
    news_plugin: Optional[BaseNewsPlugin] = None,
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
    up_feat = predict_scorelines(df_feat, up_feat, min_train_rows=min_train_rows)
    up_feat = apply_news_plugin_to_upcoming(up_feat, plugin=news_plugin or FootballNewsPlugin())
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

    extra_unplayed = raw[~raw["result"].isin(["H", "D", "A"]) & (raw["season"] == "GLOBAL")].copy()
    played_df = raw[raw["result"].isin(["H", "D", "A"]) | (raw["season"] != "GLOBAL")].copy()

    combined = played_df.copy()
    up_parts = [p for p in [upcoming_raw, extra_unplayed] if not p.empty]
    if up_parts:
        up_all = pd.concat(up_parts, ignore_index=True).drop_duplicates(subset=["match_id"], keep="last")
        fixture_status = f"{fixture_status} (+{len(extra_unplayed)} global fixture rows)" if not extra_unplayed.empty else fixture_status
        existing_ids = set(combined["match_id"].tolist())
        new_up = up_all[~up_all["match_id"].isin(existing_ids)].copy()
        if not new_up.empty:
            combined = pd.concat([combined, new_up], ignore_index=True).copy()

    feat = engineer_prematch_features(combined)
    backtest_df, brier = run_walk_forward_backtest(feat)
    acc = float((backtest_df["predicted_outcome"] == backtest_df["result"]).mean())
    roi_stats = compute_simulated_roi(backtest_df, min_edge=min_edge)

    unplayed_feat = feat[~feat["result"].isin(["H", "D", "A"])].copy()
    upcoming_scored = score_upcoming_with_latest_model(feat, unplayed_feat, news_plugin=FootballNewsPlugin())
    sb_unplayed = fetch_unplayed_from_supabase(leagues=list(leagues_tuple))
    sb_settled = fetch_settled_from_supabase(leagues=list(leagues_tuple))

    return len(raw), backtest_df, upcoming_scored, sb_unplayed, sb_settled, fixture_status, acc, brier, roi_stats


def render_streamlit_app() -> None:
    st.set_page_config(page_title="Hotshots AI Sports Engine — Global Football", layout="wide")
    _inject_sporty_theme()
    st.title("Hotshots AI Sports Engine")
    st.caption("Country/League selector • News plugin • Walk-forward LightGBM + Elo + Poisson scorelines + Kelly ROI")

    col_c, col_l, col_e = st.columns([1.2, 2.0, 1.0])
    with col_c:
        country_options = ["All Countries"] + list(FOOTBALL_LEAGUE_REGISTRY.keys())
        selected_country = st.selectbox("Country", options=country_options, index=1)

    if selected_country == "All Countries":
        available_codes = list(LEAGUE_NAMES.keys())
        default_codes = ["E0", "E1", "D1", "SP1"]
    else:
        available_codes = list(FOOTBALL_LEAGUE_REGISTRY.get(selected_country, {}).keys())
        default_codes = available_codes

    with col_l:
        selected_leagues = st.multiselect(
            "Leagues",
            options=available_codes,
            default=default_codes,
            format_func=lambda code: f"{code} — {LEAGUE_NAMES.get(code, code)}",
        )
    with col_e:
        min_edge_pct = st.slider("Min Edge (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)

    col_r, col_s = st.columns([1, 1])
    with col_r:
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col_s:
        sync_flag = st.checkbox("Sync to Supabase", value=False)

    if not selected_leagues:
        st.warning("Select at least one league to run the engine.")
        render_global_footer()
        return

    min_edge = min_edge_pct / 100.0

    try:
        with st.spinner("Loading leagues, running news plugin, Elo/rest features, and walk-forward models..."):
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

        settled_bt, kelly_stats = settle_predictions_with_kelly(backtest_df, min_edge=min_edge)

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
                deck_source = value_df if not value_df.empty else upcoming_scored.sort_values("match_date")
                render_sporty_match_deck(
                    fixtures_df=deck_source,
                    settled_df=settled_bt,
                    kelly_stats=kelly_stats,
                    model_acc=acc,
                    min_edge=min_edge,
                )

                st.divider()
                edge_cols = [
                    c
                    for c in [
                        "match_date",
                        "league_code",
                        "home_team",
                        "away_team",
                        "pred_score",
                        "news_elo_shift",
                        "best_value_side",
                        "best_edge",
                        "best_odds",
                        "prob_home",
                        "prob_draw",
                        "prob_away",
                        "news_summary",
                    ]
                    if c in deck_source.columns
                ]
                st.dataframe(deck_source[edge_cols], use_container_width=True)
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
                options=["Walk-Forward Backtest", "Supabase Settled Matches"],
                horizontal=True,
            )
            if source_choice == "Supabase Settled Matches":
                active_settled, active_kelly_stats = settle_predictions_with_kelly(sb_settled, min_edge=min_edge)
            else:
                active_settled, active_kelly_stats = settled_bt, kelly_stats

            st.subheader("Fractional Kelly Profit / Loss Curve")
            render_kelly_pnl_chart(active_settled, active_kelly_stats, height=250, compact=False)

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

    render_global_footer()


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

    print(f"Loaded {len(raw)} football matches.")
    print(f"Walk-forward evaluated matches: {len(backtest_df)}")
    print(f"Out-of-sample Accuracy: {acc:.2%} | Multi-class Brier Score: {brier:.4f}")
    print(f"Simulated ROI (>=3% edge): {roi_stats['roi']:+.2%} ({roi_stats['profit_units']:+.2f}u)")
    print(f"Synced rows to Supabase: {synced}")
