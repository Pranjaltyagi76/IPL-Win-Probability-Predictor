import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model pipeline
pipe = pickle.load(open("pipe.pkl", "rb"))

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

# Input layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    city = st.selectbox("City", cities)
    current_score = st.number_input("Current Score", min_value=0, step=1)

with col2:
    bowling_team = st.selectbox("Bowling Team", teams)
    target = st.number_input("Target Score", min_value=1, step=1)
    wickets = st.number_input("Wickets Left", min_value=0, max_value=10, step=1)

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

if runs_left < 0:
    st.error("Current score cannot be greater than target.")
    st.stop()

# Prediction
if st.button("Predict Win Probability"):

    input_df = pd.DataFrame([[
        batting_team,
        bowling_team,
        city,
        runs_left,
        balls_left,
        wickets,
        target,
        crr,
        rrr
    ]], columns=[
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets',
        'target',
        'crr',
        'rrr'
    ])

    win_prob = pipe.predict_proba(input_df)[0][1]
    lose_prob = 1 - win_prob

    # Probability output
    st.markdown("## Win Probability")
    st.progress(win_prob)

    st.success(f"{batting_team}: {win_prob * 100:.2f}%")
    st.error(f"{bowling_team}: {lose_prob * 100:.2f}%")

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

# Ending
st.markdown("---")
st.caption("Built using IPL ball-by-ball data, Machine Learning, and Streamlit")
