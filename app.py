import streamlit as st 
import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image
from torchvision import models
import sqlite3
import hashlib
import numpy as np

# Initialize session state for navigation and user role
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "login"
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None  # "admin" or "user"

# Navigation function to switch pages
def navigate_to(page_name, role=None):
    st.session_state["current_page"] = page_name
    st.session_state["user_role"] = role

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database Initialization
def init_db():
    """Initialize the SQLite database and create the users table if it doesn't exist.""" 
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(''' 
        CREATE TABLE IF NOT EXISTS users ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            username TEXT UNIQUE NOT NULL, 
            hashed_password TEXT NOT NULL, 
            role TEXT NOT NULL DEFAULT "user" 
        )
    ''')
    conn.commit()

    # Admin entry
    try:
        admin_username = "HELLFATHER4622"
        admin_password = "4622"
        c.execute("SELECT * FROM users WHERE username = ? AND role = 'admin'", (admin_username,))
        if c.fetchone() is None:  
            c.execute("INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)", 
                      (admin_username, hash_password(admin_password), "admin"))
    except Exception as e:
        st.error(f"Error adding admin user: {e}")

    conn.commit()
    conn.close()

# Database Helper Functions
def register_user(username, password, role="user"):
    """Register a new user in the database.""" 
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

def authenticate_user(username, password):
    """Authenticate a user by checking the database.""" 
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT hashed_password, role FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result and result[0] == hash_password(password):
            return result[1]  # Return the user's role
        return None
    finally:
        conn.close()

# Load Model (Using Streamlit Cache for Resource Optimization)
@st.cache_resource
def load_model():
    """Loads the trained ResNet18 model.""" 
    try:
        model = models.resnet18(pretrained=True)  
        model.fc = torch.nn.Linear(model.fc.in_features, 4) 

        # Load state_dict with key adjustments
        state_dict = torch.load("alzheimers_cnn_model.pth", map_location=torch.device('cpu'))
        state_dict = {key.replace("resnet.", "").replace("module.", ""): value for key, value in state_dict.items()}

        # Load into model
        model.load_state_dict(state_dict)
        model.eval()  
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'alzheimers_cnn_model.pth' is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction Function
def predict(image, model):
    """Preprocesses the image and predicts the class using the model.""" 
    try:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
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

# Login Page
def login_page():
    st.title("Login Page")
    st.write("Welcome back! Please log in to access the Alzheimer's Disease Classification tool.")

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = authenticate_user(username, password)
            if role:
                st.success("Login successful!")
                navigate_to("classification" if role == "user" else "admin", role)
            else:
                st.error("Invalid username or password.")

    # Link to registration page
    st.write("New to the site?")
    if st.button("Register Here"):
        navigate_to("register")

# Registration Page
def registration_page():
    st.title("Registration Page")
    st.write("Create an account to use the Alzheimer's Disease Classification tool.")

    # Registration form
    with st.form("registration_form"):
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            elif register_user(new_username, new_password):
                # Automatically log the user in after registration
                st.success("Registration successful! Logging you in...")
                role = authenticate_user(new_username, new_password)
                if role:
                    navigate_to("classification" if role == "user" else "admin", role)
                else:
                    st.error("Error during login after registration.")
            else:
                st.error("Username already exists. Please choose a different username.")

    # Back to login button
    if st.button("Back to Login"):
        navigate_to("login")

# Classification Page
def classification_page():
    st.title("Alzheimer's Disease Classification")
    st.write("Upload an image to predict the category of Alzheimer's Disease.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load the model
            model = load_model()
            if model:
                # Run Prediction
                st.write("Classifying...")
                prediction = predict(image, model)
                if prediction:
                    st.success(f"Prediction: {prediction}")
            else:
                st.error("Model could not be loaded. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # Logout button
    if st.button("Logout"):
        navigate_to("login")

# Admin Page (For Admin Users)
def admin_page():
    st.title("Admin Panel")
    st.write("Welcome to the Admin Panel. View and manage user information.")

    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT id, username, hashed_password, role FROM users")
        users = c.fetchall()

        # Display user data in a table
        import pandas as pd
        df = pd.DataFrame(users, columns=["ID", "Username", "Hashed Password", "Role"])
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    finally:
        conn.close()

    # Logout button
    if st.button("Logout"):
        navigate_to("login")

# Page Router
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

# Initialize the database
init_db()
