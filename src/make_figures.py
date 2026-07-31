"""Generate the figures embedded in the README.

Kept as a script so every image in the docs is reproducible from the data and
the shipped model rather than being an undocumented binary.

Run from anywhere:  python src/make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict import load_model, build_features, explain
from replay import load_data, win_probability_curve, FEATURED_MATCH_ID
from train_model import DEFAULT_MATCHES, DEFAULT_DELIVERIES

REPORTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "reports")

# The README renders on either a light or a dark GitHub theme, so the figure
# bakes in its own dark background instead of being transparent.
PAGE_BG = "#0e1117"

STYLE = {
    "figure.facecolor": PAGE_BG,
    "axes.facecolor": PAGE_BG,
    "savefig.facecolor": PAGE_BG,
    "text.color": "#fafafa",
    "axes.labelcolor": "#fafafa",
    "axes.titlecolor": "#fafafa",
    "axes.edgecolor": "#8b93a7",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "legend.facecolor": "#1a1f2b",
    "legend.edgecolor": "#2a3040",
    "legend.labelcolor": "#fafafa",
}


def replay_figure(pipe, matches, deliveries, match_id=FEATURED_MATCH_ID):
    """The ball-by-ball win-probability curve, as shown in the app."""
    curve = win_probability_curve(pipe, match_id, matches, deliveries)
    info = matches[matches["match_id"] == match_id].iloc[0]
    chaser = deliveries[
        (deliveries["match_id"] == match_id) & (deliveries["innings"] == 2)
    ]["batting_team"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(curve["ball_no"], curve["win_prob"] * 100,
            color="#4c9be8", linewidth=2, zorder=2)
    ax.fill_between(curve["ball_no"], curve["win_prob"] * 100,
                    color="#4c9be8", alpha=0.15, zorder=1)
    ax.axhline(50, color="#8b93a7", linestyle="--", linewidth=1, zorder=1)

    fours = curve[(curve["is_boundary"] == 1) & (curve["runs_this_ball"] == 4)]
    sixes = curve[(curve["is_boundary"] == 1) & (curve["runs_this_ball"] == 6)]
    wickets = curve[curve["is_wicket"] == 1]

    ax.scatter(fours["ball_no"], fours["win_prob"] * 100,
               color="#2ca02c", marker="o", s=28, zorder=3, label="Four")
    ax.scatter(sixes["ball_no"], sixes["win_prob"] * 100,
               color="#b07aff", marker="^", s=45, zorder=3, label="Six")
    ax.scatter(wickets["ball_no"], wickets["win_prob"] * 100,
               color="#ff4b4b", marker="v", s=70, zorder=4, label="Wicket")

    ax.legend(loc="lower left", ncol=3, fontsize=8, framealpha=0.9)
    ax.set_title(
        f"{info['team1']} vs {info['team2']} — IPL {info['season']} "
        f"({info['city']})",
        fontsize=11,
    )
    ax.set_xlabel("Ball of the second innings")
    ax.set_ylabel(f"{chaser} win probability (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(curve["ball_no"].min(), curve["ball_no"].max())
    fig.tight_layout()
    return fig


def explanation_figure(pipe, matches, deliveries, match_id=FEATURED_MATCH_ID):
    """Why the model made its lowest call of the featured chase.

    Takes the single most pessimistic ball of the match and decomposes that
    prediction into per-feature contributions, which is what the app shows
    under "Why this prediction".
    """
    curve = win_probability_curve(pipe, match_id, matches, deliveries)
    match = matches[matches["match_id"] == match_id].iloc[0]
    low = curve.loc[curve["win_prob"].idxmin()]

    chase = deliveries[
        (deliveries["match_id"] == match_id) & (deliveries["innings"] == 2)
    ]
    batting = chase["batting_team"].iloc[0]
    bowling = chase["bowling_team"].iloc[0]

    row = build_features(
        batting, bowling, match["city"],
        int(low["runs_left"]), int(low["balls_left"]), int(low["wickets"]),
        int(match["target"]), float(low["crr"]), float(low["rrr"]),
    )
    contributions = explain(pipe, row)

    labels = [c[0] for c in contributions][::-1]
    values = [c[1] for c in contributions][::-1]
    colours = ["#2ca02c" if v > 0 else "#ff4b4b" for v in values]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(labels, values, color=colours)
    ax.axvline(0, color="#8b93a7", linewidth=0.9)
    ax.set_xlabel("Contribution to win probability (log-odds)")
    # Two lines, so the long categorical labels can't push the title off-canvas.
    ax.set_title(
        f"Why {batting} were rated {low['win_prob'] * 100:.0f}%\n"
        f"{int(low['runs_left'])} needed off {int(low['balls_left'])} balls, "
        f"{int(low['wickets'])} wickets in hand",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    pipe = load_model()
    matches, deliveries = load_data(DEFAULT_MATCHES, DEFAULT_DELIVERIES)

    figures = {
        "replay_featured.png": replay_figure(pipe, matches, deliveries),
        "explanation_featured.png": explanation_figure(
            pipe, matches, deliveries
        ),
    }
    for name, fig in figures.items():
        out = os.path.join(REPORTS_DIR, name)
        fig.savefig(out, dpi=130)
        print("Saved", os.path.normpath(out))


if __name__ == "__main__":
    main()
