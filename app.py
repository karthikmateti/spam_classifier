import streamlit as st
import pickle

# Load model
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📧 Email Spam Classifier")

st.write("Check whether an email is Spam or Ham")

# Input box
user_input = st.text_area("Enter your email text:")

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
