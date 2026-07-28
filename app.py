import os
import sys

import streamlit as st
import matplotlib.pyplot as plt

# Match the dark Streamlit theme: transparent figure backgrounds with light
# axes and text, so charts blend into the page instead of sitting inside
# white boxes.
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": "#fafafa",
    "axes.labelcolor": "#fafafa",
    "axes.titlecolor": "#fafafa",
    "axes.edgecolor": "#8b93a7",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "legend.facecolor": "#1a1f2b",
    "legend.edgecolor": "#2a3040",
    "legend.labelcolor": "#fafafa",
})

# Inference + replay helpers live in src/; make them importable from the root.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model, build_features, win_probability, explain
from replay import (
    load_data, list_matches, win_probability_curve, FEATURED_MATCH_ID
)

# Load trained model pipeline
pipe = load_model()

# Page configuration
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    layout="centered"
)

st.title("IPL Win Probability Predictor")

# Team and city lists
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi',
    'Chandigarh','Jaipur','Chennai','Cape Town','Port Elizabeth',
    'Durban','Centurion','East London','Johannesburg','Kimberley',
    'Bloemfontein','Ahmedabad','Cuttack','Nagpur','Dharamsala',
    'Visakhapatnam','Pune','Raipur','Ranchi','Abu Dhabi',
    'Sharjah','Mohali','Bengaluru'
]


def render_manual():
    """Manual mode: enter a hypothetical match state and predict."""
    # Input layout
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team", teams)
        city = st.selectbox("City", cities)
        current_score = st.number_input("Current Score", min_value=0, step=1)

    with col2:
        bowling_team = st.selectbox("Bowling Team", teams)
        target = st.number_input("Target Score", min_value=1, step=1)
        wickets = st.number_input(
            "Wickets Left", min_value=0, max_value=10, step=1
        )

    balls_left = st.slider("Balls Left", min_value=1, max_value=120)

    # Feature engineering
    runs_left = target - current_score

    overs_played = (120 - balls_left) / 6
    overs_left = balls_left / 6

    crr = (current_score / overs_played) if overs_played > 0 else 0
    rrr = (runs_left / overs_left) if overs_left > 0 else 0

    # Run rate display
    st.info(f"Current Run Rate (CRR): {crr:.2f}")
    st.info(f"Required Run Rate (RRR): {rrr:.2f}")

    # Match summary
    st.markdown("### Match Summary")
    st.write(f"""
- Batting Team: {batting_team}
- Bowling Team: {bowling_team}
- City: {city}
- Target: {target}
- Runs Needed: {runs_left}
- Balls Remaining: {balls_left}
- Wickets Left: {wickets}
""")

    # Validations
    if batting_team == bowling_team:
        st.error("Batting and bowling teams must be different.")
        st.stop()

    # current_score >= target is not an error — it means the chase is already
    # won. That case is resolved at prediction time, so we don't stop here.

    # Prediction
    if st.button("Predict Win Probability"):

        input_df = build_features(
            batting_team, bowling_team, city,
            runs_left, balls_left, wickets, target, crr, rrr
        )

        win_prob, status = win_probability(pipe, input_df)
        if status:
            st.info(status)

        lose_prob = 1 - win_prob

        # Probability output
        st.markdown("## Win Probability")
        st.progress(win_prob)

        st.success(f"{batting_team}: {win_prob * 100:.2f}%")
        st.error(f"{bowling_team}: {lose_prob * 100:.2f}%")

        # Explanation: exact per-feature contributions to the win log-odds.
        # Only meaningful when the model actually made the call (not a forced
        # all-out / target-reached certainty).
        if status is None:
            st.markdown("### Why this prediction")
            contribs = explain(pipe, input_df)

            # Plain-English summary of the biggest factors either way.
            helping = [label for label, c in contribs if c > 0][:3]
            hurting = [label for label, c in contribs if c < 0][:3]
            if helping:
                st.markdown(
                    f"**Helping {batting_team}:** " + ", ".join(helping)
                )
            if hurting:
                st.markdown(
                    f"**Hurting {batting_team}:** " + ", ".join(hurting)
                )

            labels = [c[0] for c in contribs][::-1]
            values = [c[1] for c in contribs][::-1]
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

            figx, axx = plt.subplots(figsize=(7, 3.5))
            axx.barh(labels, values, color=colors)
            axx.axvline(0, color="black", linewidth=0.8)
            axx.set_xlabel("Contribution to win probability (log-odds)")
            st.pyplot(figx)

        # Graph 1: CRR vs RRR
        st.markdown("### Run Rate Comparison")

        fig1, ax1 = plt.subplots()
        ax1.bar(["Current Run Rate", "Required Run Rate"], [crr, rrr])
        ax1.set_ylabel("Runs per Over")
        ax1.set_title("CRR vs RRR")

        st.pyplot(fig1)

        # Graph 2: Win probability bar
        st.markdown("### Team-wise Win Probability")

        fig2, ax2 = plt.subplots()
        ax2.bar(
            [batting_team, bowling_team],
            [win_prob * 100, lose_prob * 100]
        )
        ax2.set_ylabel("Win Probability (%)")
        ax2.set_ylim(0, 100)

        st.pyplot(fig2)

        # Graph 3: Match pressure snapshot
        st.markdown("### Match Pressure Snapshot")

        fig3, ax3 = plt.subplots()
        ax3.scatter(balls_left, runs_left, color="red", s=100)
        ax3.set_xlabel("Balls Left")
        ax3.set_ylabel("Runs Left")
        ax3.set_title("Runs Required vs Balls Remaining")

        st.pyplot(fig3)


def render_replay():
    """Replay mode: pick a real historical chase to analyse ball by ball."""
    st.caption(
        "Pick a real IPL chase and see how the win probability moved across "
        "the second innings."
    )

    matches, deliveries = load_data(
        os.path.join("data", "matches.csv"),
        os.path.join("data", "deliveries.csv"),
    )
    catalogue = list_matches(matches, deliveries)
    labels = catalogue["label"].tolist()

    # Open on the featured thriller (the 2008 final) if it is available.
    featured = catalogue.index[catalogue["id"] == FEATURED_MATCH_ID]
    default_index = int(featured[0]) if len(featured) else 0

    choice = st.selectbox("Select a match", labels, index=default_index)
    match_id = int(catalogue.loc[catalogue["label"] == choice, "id"].iloc[0])

    info = matches[matches["id"] == match_id].iloc[0]
    if info["win_by_runs"] > 0:
        margin = f"won by {int(info['win_by_runs'])} runs"
    elif info["win_by_wickets"] > 0:
        margin = f"won by {int(info['win_by_wickets'])} wickets"
    else:
        margin = "result decided without a standard margin"

    st.markdown("### Match Summary")
    st.write(f"""
- Season: {info['Season']}
- Teams: {info['team1']} vs {info['team2']}
- Venue: {info['city']}
- Result: {info['winner']} {margin}
- Player of the Match: {info['player_of_match']}
""")

    # Ball-by-ball win-probability curve for the chasing team.
    curve = win_probability_curve(pipe, match_id, matches, deliveries)
    if curve.empty:
        st.warning("No second-innings data available for this match.")
        return

    chaser = deliveries[
        (deliveries["match_id"] == match_id) & (deliveries["inning"] == 2)
    ]["batting_team"].iloc[0]

    st.markdown(f"### Win Probability — {chaser} (chasing)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curve["ball_no"], curve["win_prob"] * 100,
            color="#1f77b4", linewidth=2, zorder=2)
    ax.fill_between(curve["ball_no"], curve["win_prob"] * 100,
                    color="#1f77b4", alpha=0.12, zorder=1)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, zorder=1)

    # Boundaries lift the curve; sixes are called out separately from fours.
    fours = curve[(curve["is_boundary"] == 1) & (curve["runs_this_ball"] == 4)]
    sixes = curve[(curve["is_boundary"] == 1) & (curve["runs_this_ball"] == 6)]
    if not fours.empty:
        ax.scatter(fours["ball_no"], fours["win_prob"] * 100,
                   color="#2ca02c", marker="o", s=28, zorder=3, label="Four")
    if not sixes.empty:
        ax.scatter(sixes["ball_no"], sixes["win_prob"] * 100,
                   color="#9467bd", marker="^", s=45, zorder=3, label="Six")

    # Mark every wicket — these are the sharpest swings in the curve.
    wickets_hit = curve[curve["is_wicket"] == 1]
    if not wickets_hit.empty:
        ax.scatter(wickets_hit["ball_no"], wickets_hit["win_prob"] * 100,
                   color="#d62728", marker="v", s=70, zorder=4,
                   label="Wicket")

    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax.set_xlabel("Ball of the second innings")
    ax.set_ylabel(f"{chaser} win probability (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(curve["ball_no"].min(), curve["ball_no"].max())
    st.pyplot(fig)

    st.caption("▼ wicket · ● four · ▲ six · dashed line = 50% (even contest)")

    # Drama highlights computed straight from the curve.
    p = curve["win_prob"].values
    lead_changes = int(((p[:-1] - 0.5) * (p[1:] - 0.5) < 0).sum())
    chaser_won = info["winner"] == chaser
    if chaser_won:
        swing = f"recovered from as low as {p.min() * 100:.0f}%"
    else:
        swing = f"peaked at {p.max() * 100:.0f}% but fell short"

    c1, c2 = st.columns(2)
    c1.metric("Lead changes (50% crossings)", lead_changes)
    c2.metric(f"{chaser} (chasing)", swing)


# --- Mode selector -----------------------------------------------------------
mode = st.sidebar.radio(
    "Mode", ["Match Replay", "Manual Prediction"]
)

if mode == "Match Replay":
    render_replay()
else:
    render_manual()

# Ending
st.markdown("---")
st.caption("Built using IPL ball-by-ball data, Machine Learning, and Streamlit")
