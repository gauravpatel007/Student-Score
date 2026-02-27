#python -m streamlit run app.py
import time
# from altair import value
import numpy as np
# from sqlalchemy import label
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student ML Dashboard",
                   layout="wide",
                   page_icon="🎓")

# ---------------- THEME SWITCH ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme_choice = st.sidebar.toggle("🌙 Dark Mode")

st.session_state.theme = "dark" if theme_choice else "light"
# ---------------- THEME CSS ----------------

if st.session_state.theme == "dark":
    st.markdown("""
<style>
/* General font for both modes */
body, .stApp, .stMarkdown {
    font-family: 'Segoe UI', sans-serif;
}

/* Cards */
.card {
    padding:20px;
    border-radius:16px;
    box-shadow:0 4px 12px rgba(0,0,0,0.12);
    transition: all 0.25s ease;
}

/* Card hover effect */
.card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow:0 12px 24px rgba(0,0,0,0.18);
}

/* Small text inside cards */
.small {
    font-size:14px;
    font-weight:500;
}

/* Metric number inside cards */
.metric {
    font-size:28px;
    font-weight:700;
}

/* Dark mode */
body[data-theme="dark"] .stApp, .dark-theme {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
}
body[data-theme="dark"] .card, .dark-theme .card {
    background-color: #1e293b !important;
    color: #e5e7eb !important;
}

/* Light mode */
body[data-theme="light"] .stApp, .light-theme {
    background-color: #f9fafb !important;
    color: #111827 !important;
}
body[data-theme="light"] .card, .light-theme .card {
    background-color: white !important;
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)


else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f9fafb;
        color: #111827;
    }
    .card {
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:700;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.card {
    padding:20px;
    border-radius:16px;
    box-shadow:0 4px 12px rgba(0,0,0,0.12);
    background:white;
    transition: all 0.25s ease;
}
.metric {
    font-size:28px;
    font-weight:700;
}
.small {
    color:gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users(u TEXT,p TEXT)")
c.execute("""CREATE TABLE IF NOT EXISTS history(
u TEXT, study REAL, attend REAL, assign REAL, prev REAL, pred REAL)""")
conn.commit()

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("student_score_big_dataset.csv")

data = load_data()

# ---------------- MODEL ----------------
features = ["StudyHours","Attendance","Assignments","PreviousScore"]
X = data[features]
y = data["Score"]

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    model = LinearRegression().fit(X,y)
    joblib.dump(model,"model.pkl")

# ---------------- AUTH ----------------
def login(u,p):
    c.execute("SELECT * FROM users WHERE u=? AND p=?",(u,p))
    return c.fetchone()

def register(u,p):
    try:
        c.execute("INSERT INTO users VALUES (?,?)",(u,p))
        conn.commit()
        return True
    except:
        return False

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎓 Student Performance ML Dashboard</div>', unsafe_allow_html=True)
st.write("Predict • Analyze • Visualize")

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("Navigation",
    ["Login","Register","Dashboard"])

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- REGISTER ----------------
if menu == "Register":
    st.subheader("Create Account")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Create Account"):
        if register(u,p):
            st.success("Account created")
        else:
            st.error("User exists")

# ---------------- LOGIN ----------------
if menu == "Login":
    st.subheader("User Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u,p):
            st.session_state.user = u
            st.success("Welcome " + u)
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
if menu == "Dashboard" and st.session_state.user:

    # ===== KPI CARDS =====
    def animated_metric(label, value, suffix=""):
        placeholder = st.empty()
        for i in np.linspace(0, value, 30):
            placeholder.markdown(
                f'<div class="card"><div class="small">{label}</div>'
                f'<div class="metric">{i:.1f}{suffix}</div></div>',
                unsafe_allow_html=True
            )
            time.sleep(0.01)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        animated_metric("Records", len(data))
    with col2:
        animated_metric("Avg Score", data.Score.mean())
    with col3:
        animated_metric("Avg Study Hours", data.StudyHours.mean())
    with col4:
        animated_metric("Avg Attendance", data.Attendance.mean(), "%")

    #****************************************
    chart_style = st.sidebar.selectbox(
        "Chart Style",
        ["default","ggplot","bmh","seaborn-v0_8"]
    )
    plt.style.use(chart_style)
    #****************************************
    tab1,tab2,tab3 = st.tabs(
        ["📊 Analytics","🤖 Prediction","📜 History"]
    )
    # -------- ANALYTICS TAB --------
    with tab1:
        c1,c2 = st.columns(2)

        with c1:
            st.subheader("Study Hours vs Score")
            fig = plt.figure()
            plt.scatter(data.StudyHours, data.Score)
            plt.xlabel("Study Hours")
            plt.ylabel("Score")
            st.pyplot(fig)

        with c2:
            st.subheader("Attendance Distribution")
            fig2 = plt.figure()
            plt.hist(data.Attendance, bins=20)
            st.pyplot(fig2)

        st.subheader("Correlation Matrix")
        st.dataframe(data.corr())

    # -------- PREDICTION TAB --------
    with tab2:
        st.subheader("Enter Student Inputs")

        c1,c2 = st.columns(2)
        study = c1.slider("Study Hours",0,15,5)
        attendance = c2.slider("Attendance %",0,100,75)
        assignments = c1.slider("Assignments",0,12,6)
        previous = c2.slider("Previous Score",0,100,60)

        if st.button("🚀 Predict Score"):
            inp = pd.DataFrame([[study,attendance,assignments,previous]],columns=features)
            pred = model.predict(inp)[0]

            st.markdown(
                f'<div class="card"><div class="small">Predicted Score</div>'
                f'<div class="metric">{pred:.2f}</div></div>',
                unsafe_allow_html=True)

            st.subheader("Score Confidence Meter")

            progress = st.progress(0)
            for i in range(int(pred)):
                progress.progress(i+1)
                time.sleep(0.01)

            if pred >= 80:
                st.success("Excellent Performance Zone")
            elif pred >= 60:
                st.info("Good Performance Zone")
            else:
                st.warning("Needs Improvement Zone")    

    # -------- HISTORY TAB --------
    with tab3:
        st.subheader("Prediction History")
        df = pd.read_sql_query(
            "SELECT * FROM history WHERE u=?",
            conn,
            params=(st.session_state.user,))
        st.dataframe(df)

elif menu == "Dashboard":
    st.warning("Please login first")
