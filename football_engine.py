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

    df = df.sort_values(["match_date_dt", "league_code", "match_id"]).reset_index(drop=True).copy()
    df = df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True).copy()
    return df
