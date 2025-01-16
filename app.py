import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import sqlite3
import hashlib
import datetime
import pandas as pd

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "login"
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# Navigation function
def navigate_to(page_name, role=None, username=None):
    st.session_state["current_page"] = page_name
    st.session_state["user_role"] = role
    st.session_state["username"] = username

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT "user"
        )
    ''')

    # Sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            activity TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT 1
        )
    ''')

    conn.commit()
    conn.close()

# Register user
def register_user(username, password, role="user"):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                  (username, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT hashed_password, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return result[1]
    return None

# Track login sessions
def log_login(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO sessions (username, is_active) VALUES (?, 1)", (username,))
    conn.commit()
    conn.close()

# Log activity
def log_activity(username, activity):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE sessions SET activity = ?, login_time = CURRENT_TIMESTAMP WHERE username = ? AND is_active = 1", (activity, username))
    conn.commit()
    conn.close()

# End session
def end_session(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE sessions SET is_active = 0 WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# Deregister user
def deregister_user(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        c.execute("DELETE FROM sessions WHERE username = ?", (username,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deregistering user: {e}")
        return False
    finally:
        conn.close()

# Load model
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)

        state_dict = torch.load("alzheimers_cnn_model.pth", map_location=torch.device('cpu'))
        state_dict = {key.replace("resnet.", "").replace("module.", ""): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file not found.")
        return None

# Predict
def predict(image, model):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0)
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
        class_labels = [
            "AD (Alzheimer's Disease)",
            "CN (Cognitively Normal)",
            "EMCI (Early Mild Cognitive Impairment)",
            "LMCI (Late Mild Cognitive Impairment)"
        ]
        return class_labels[predicted_class.item()]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Login page
def login_page():
    st.title("Login Page")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = authenticate_user(username, password)
            if role:
                st.success("Login successful!")
                log_login(username)
                navigate_to("classification" if role == "user" else "admin", role, username)
            else:
                st.error("Invalid username or password.")

    if st.button("Register Here"):
        navigate_to("register")

# Registration page
def registration_page():
    st.title("Registration Page")

    with st.form("registration_form"):
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif register_user(new_username, new_password):
                st.success("Registration successful! Redirecting to classification page.")
                log_activity(new_username, "Registered successfully")
                navigate_to("classification", role="user", username=new_username)
            else:
                st.error("Username already exists.")

    if st.button("Back to Login"):
        navigate_to("login")

# Classification page
def classification_page():
    st.title("Alzheimer's Disease Classification")
    log_activity(st.session_state["username"], "Accessed classification page")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            model = load_model()
            if model:
                prediction = predict(image, model)
                if prediction:
                    st.success(f"Prediction: {prediction}")
                    log_activity(st.session_state["username"], f"Predicted: {prediction}")
            else:
                st.error("Model could not be loaded.")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Logout"):
        end_session(st.session_state["username"])
        st.session_state.clear()
        navigate_to("login")

# Admin page
def admin_page():
    st.title("Admin Panel")

    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT username, login_time, activity, is_active FROM sessions")
        sessions = c.fetchall()

        c.execute("SELECT username, role FROM users")
        users = c.fetchall()

        df_sessions = pd.DataFrame(sessions, columns=["Username", "Login Time", "Activity", "Is Active"])
        st.subheader("Active Sessions")
        st.dataframe(df_sessions)

        df_users = pd.DataFrame(users, columns=["Username", "Role"])
        st.subheader("Registered Users")
        st.dataframe(df_users)

        # Deregister user feature
        with st.form("deregister_user_form"):
            username_to_remove = st.text_input("Enter username to deregister")
            submitted = st.form_submit_button("Deregister User")

            if submitted:
                if deregister_user(username_to_remove):
                    st.success(f"User '{username_to_remove}' has been deregistered successfully.")
                else:
                    st.error(f"Failed to deregister user '{username_to_remove}'.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        conn.close()

    if st.button("Logout"):
        st.session_state.clear()
        navigate_to("login")

# Page router
if st.session_state["current_page"] == "login":
    login_page()
elif st.session_state["current_page"] == "register":
    registration_page()
elif st.session_state["current_page"] == "classification" and st.session_state["user_role"] == "user":
    classification_page()
elif st.session_state["current_page"] == "admin" and st.session_state["user_role"] == "admin":
    admin_page()
else:
    st.error("Access Denied!")

init_db()
