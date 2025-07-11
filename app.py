import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

# ✅ Replaced st.beta_columns() with st.columns()
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = int(st.number_input('Enter Target', step=1, format="%d"))

col3, col4, col5 = st.columns(3)

with col3:
    score = int(st.number_input('Score', step=1, format="%d"))
with col4:
    full_overs = st.number_input('Full Overs Completed', min_value=0, step=1, format="%d")
    extra_balls = st.number_input('Balls into current over (0–5)', min_value=0, max_value=5, step=1, format="%d")

# Calculate actual overs as float
overs = full_overs + (extra_balls / 6)

with col5:
    wickets = int(st.number_input('Wickets out', min_value=0, max_value=10, step=1, format="%d"))


if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs != 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
