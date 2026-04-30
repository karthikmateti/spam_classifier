import streamlit as st
import pickle
import os

# -------------------------------
# Sidebar (Project Info)
# -------------------------------
st.sidebar.title("📌 ML PROJECT")

st.sidebar.markdown("""
### 👨‍💻 Team Members
- Avaneesh Reddy
- Mateti Karthik

### 📚 Course
Machine Learning  

### 👨‍🏫 Instructor
Mariya Celin T 
""")

# -------------------------------
# Load Model Safely
# -------------------------------
if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error("Model files not found. Please check deployment.")
else:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    # -------------------------------
    # Main UI
    # -------------------------------
    st.title("📧 Email Spam Classifier")

    st.write("Check whether an email is Spam or Ham")

    # Input box
    user_input = st.text_area("Enter your email text:")

    # Predict button
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            input_data = [user_input]
            vector_input = vectorizer.transform(input_data)
            prediction = model.predict(vector_input)

            if prediction[0] == 1:
                st.success("✅ Ham Mail (Not Spam)")
            else:
                st.error("🚨 Spam Mail")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Avaneesh & Karthik| ML Project 🚀")
