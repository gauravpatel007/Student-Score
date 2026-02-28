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

font_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Outfit', sans-serif !important;
}

/* Base gradients and styles */
:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
}

.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.2;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    animation: glowFade 2s ease-in-out infinite alternate;
}

.small {
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    margin-bottom: 8px;
}

.metric {
    font-size: 2.8rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* Animations */
@keyframes fadeUp {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes glowFade {
    0% { text-shadow: 0 0 10px rgba(99, 102, 241, 0.1); }
    100% { text-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 30px rgba(236, 72, 153, 0.2); }
}

/* Buttons */
button[kind="primary"] {
    background: var(--primary-gradient) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

/* Fix tabs */
[data-baseweb="tab-list"] {
    gap: 20px;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}
"""

if st.session_state.theme == "dark":
    theme_css = """
    .stApp {
        background-color: #0f172a !important;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
        color: #f8fafc !important;
    }
    
    .card {
        padding: 24px;
        border-radius: 20px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        color: #f8fafc;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeUp 0.6s ease-out forwards;
    }
    
    .card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 20px rgba(99, 102, 241, 0.3);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
    }
    """
else:
    theme_css = """
    .stApp {
        background-color: #f3f4f6 !important;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,97%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,100%,89%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,100%,89%,1) 0, transparent 50%);
        color: #1f2937 !important;
    }
    
    .card {
        padding: 24px;
        border-radius: 20px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        color: #1f2937;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeUp 0.6s ease-out forwards;
    }

    .card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 15px 35px rgba(31, 38, 135, 0.2), 0 0 20px rgba(99, 102, 241, 0.2);
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        background: rgba(255,255,255,0.7) !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
    }
    """

st.markdown(f"<style>{font_css}{theme_css}</style>", unsafe_allow_html=True)

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
        ["🤖 Prediction","📜 History","📊 Analytics"]
    )
    
    # -------- PREDICTION TAB --------
    with tab1:
        st.subheader("Enter Student Inputs")

        c1,c2 = st.columns(2)
        study = c1.slider("Study Hours",0,15,5)
        attendance = c2.slider("Attendance %",0,100,75)
        assignments = c1.slider("Assignments",0,12,6)
        previous = c2.slider("Previous Score",0,100,60)

        if st.button("🚀 Predict Score"):
            inp = pd.DataFrame([[study,attendance,assignments,previous]],columns=features)
            pred = model.predict(inp)[0]
            
            # Ensure predicted score stays within 0-100 bounds
            pred = min(100.0, max(0.0, pred))
            
            # Save prediction to history table
            c.execute("INSERT INTO history VALUES (?,?,?,?,?,?)", 
                      (st.session_state.user, study, attendance, assignments, previous, pred))
            conn.commit()

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
    with tab2:
        st.subheader("Prediction History")
        df = pd.read_sql_query(
            "SELECT * FROM history WHERE u=?",
            conn,
            params=(st.session_state.user,))
        st.dataframe(df)

    # -------- ANALYTICS TAB --------
    with tab3:
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

elif menu == "Dashboard":
    st.warning("Please login first")
