"""Build the canonical IPL dataset from Cricsheet.

The original project used a Kaggle export that stops at the 2019 season. This
script pulls the ball-by-ball archive from Cricsheet instead, which is free,
needs no credentials, and is updated through the current season.

It produces two files in data/:

  matches_all.csv       one row per match (result, venue, target, DLS flag)
  deliveries_all.csv.gz one row per delivery (gzipped; pandas reads it directly)

Only the columns the model needs are kept, which is what makes the delivery
table small enough to commit.

Run from anywhere:  python src/build_dataset.py
"""

import argparse
import csv
import io
import os
import urllib.request
import zipfile

import pandas as pd

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"

_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
DATA_DIR = os.path.join(_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ZIP_PATH = os.path.join(RAW_DIR, "ipl_csv2.zip")

MATCHES_OUT = os.path.join(DATA_DIR, "matches_all.csv")
DELIVERIES_OUT = os.path.join(DATA_DIR, "deliveries_all.csv.gz")

# 'retired hurt' is not a dismissal — the batter may return, and no wicket is
# credited to the bowling side. Every other wicket_type costs a wicket.
NON_DISMISSALS = {"retired hurt"}

# Franchises that were renamed but are continuous entities. Collapsing these
# keeps a single team history instead of splitting one side across two labels.
#
# Deliberately NOT included: Deccan Chargers, Gujarat Lions, Kochi Tuskers
# Kerala and Pune Warriors. Those franchises were terminated rather than
# renamed — Sunrisers Hyderabad and Gujarat Titans are separate entities that
# later occupied the same cities — so they stay distinct. The one-hot encoder
# handles the defunct labels, and the model is free to learn that they behaved
# differently.
TEAM_RENAMES = {
    "Delhi Daredevils": "Delhi Capitals",                    # renamed 2019
    "Kings XI Punjab": "Punjab Kings",                        # renamed 2021
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",  # 2024
    "Rising Pune Supergiants": "Rising Pune Supergiant",      # spelling, 2017
}


# Some matches carry no city in the source — the 2014 and 2020 UAE legs, where
# 51 matches were played at these two grounds. The city is recoverable from the
# venue name, and without it those matches drop out of training entirely.
VENUE_CITIES = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
}


def normalise_teams(df, columns):
    """Apply the franchise rename map to the given team-name columns."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].replace(TEAM_RENAMES)
    return df


def download(url=CRICSHEET_URL, dest=ZIP_PATH, force=False):
    """Fetch the Cricsheet archive unless it is already cached."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force:
        print(f"Using cached archive: {os.path.normpath(dest)}")
        return dest

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"Saved {os.path.normpath(dest)} ({size_mb:.1f} MB)")
    return dest


def parse_info(raw_bytes):
    """Parse one Cricsheet *_info.csv into a flat dict.

    The format is one 'info,<key>,<value>[,<value2>]' record per line, with
    some keys repeating (team, player, umpire).
    """
    out = {"teams": []}
    text = raw_bytes.decode("utf-8", errors="replace")

    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3 or row[0] != "info":
            continue
        key, value = row[1], row[2]

        # Some records are present but blank ('info,city,'). Treat those as
        # missing, otherwise an empty string survives as a real value and
        # defeats the fallbacks below.
        if value == "" and key != "team":
            continue

        if key == "team":
            out["teams"].append(value)
        elif key == "target_runs" and len(row) >= 4:
            # 'info,target_runs,<innings>,<runs>'
            out["target"] = int(row[3])
        elif key in {"city", "venue", "winner", "player_of_match", "method",
                     "outcome", "eliminator", "date", "season"}:
            out.setdefault(key, value)
        elif key in {"winner_runs", "winner_wickets"}:
            out[key] = value

    return out


def build_matches(archive):
    """One row per match, from the per-match info files."""
    rows = []
    with zipfile.ZipFile(archive) as z:
        info_names = [n for n in z.namelist() if n.endswith("_info.csv")]
        for name in info_names:
            info = parse_info(z.read(name))
            match_id = int(name.split("_")[0])
            teams = info.get("teams", [])

            rows.append({
                "match_id": match_id,
                "date": info.get("date"),
                "city": info.get("city"),
                "venue": info.get("venue"),
                "team1": teams[0] if len(teams) > 0 else None,
                "team2": teams[1] if len(teams) > 1 else None,
                "winner": info.get("winner"),
                "target": info.get("target"),
                # A revised target means the match was affected by rain rules.
                "dls": 1 if info.get("method") else 0,
                # 'no result' / 'tie' when there is no outright winner.
                "outcome": info.get("outcome"),
                "eliminator": info.get("eliminator"),
                # Victory margin: exactly one of these is set for a result.
                "win_by_runs": info.get("winner_runs"),
                "win_by_wickets": info.get("winner_wickets"),
                "player_of_match": info.get("player_of_match"),
            })

    matches = pd.DataFrame(rows)
    matches["date"] = pd.to_datetime(
        matches["date"].str.replace("/", "-"), errors="coerce"
    )
    # Season labels in the source are inconsistent ('2007/08', '2020/21'), but
    # each maps to exactly one calendar year, so derive it from the date.
    matches["season"] = matches["date"].dt.year
    matches["city"] = matches["city"].fillna(
        matches["venue"].map(VENUE_CITIES)
    )
    matches = normalise_teams(matches, ["team1", "team2", "winner"])
    return matches.sort_values("match_id").reset_index(drop=True)


def build_deliveries(archive):
    """One row per delivery, reduced to the columns the model needs."""
    with zipfile.ZipFile(archive) as z:
        with z.open("all_matches.csv") as f:
            df = pd.read_csv(f, low_memory=False)

    out = pd.DataFrame({
        "match_id": df["match_id"].astype("int64"),
        "innings": df["innings"].astype("int16"),
        "ball": df["ball"].astype("float32"),
        "batting_team": df["batting_team"],
        "bowling_team": df["bowling_team"],
        "batsman_runs": df["runs_off_bat"].fillna(0).astype("int16"),
        "total_runs": (
            df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
        ).astype("int16"),
    })

    # Wides and no-balls do not consume a legal delivery, so they must not
    # advance the balls-bowled count.
    out["is_legal"] = (
        df["wides"].isna() & df["noballs"].isna()
    ).astype("int8")

    out["is_wicket"] = (
        df["player_dismissed"].notna()
        & ~df["wicket_type"].isin(NON_DISMISSALS)
    ).astype("int8")

    return normalise_teams(out, ["batting_team", "bowling_team"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true",
                        help="re-download the archive even if cached")
    args = parser.parse_args()

    archive = download(force=args.force_download)

    matches = build_matches(archive)
    deliveries = build_deliveries(archive)

    os.makedirs(DATA_DIR, exist_ok=True)
    matches.to_csv(MATCHES_OUT, index=False)
    deliveries.to_csv(DELIVERIES_OUT, index=False, compression="gzip")

    print(f"\nmatches:    {len(matches):,} rows -> "
          f"{os.path.basename(MATCHES_OUT)} "
          f"({os.path.getsize(MATCHES_OUT) / 1e6:.1f} MB)")
    print(f"deliveries: {len(deliveries):,} rows -> "
          f"{os.path.basename(DELIVERIES_OUT)} "
          f"({os.path.getsize(DELIVERIES_OUT) / 1e6:.1f} MB)")
    print(f"seasons:    {matches['season'].min()}-{matches['season'].max()}")


if __name__ == "__main__":
    main()
