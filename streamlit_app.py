import streamlit as st
import joblib

model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tf-idfvector.pkl')

st.title("Email Spam Classification")

mail_text = st.text_area("Enter the email text:")

if st.button("Classify"):
    if mail_text:
        mail_tfidf = vectorizer.transform([mail_text])
        prediction = model.predict(mail_tfidf)                
        if prediction[0] == 1:
            st.error("SPAM ⚠️")
        else:
            st.success("HAM!")            
    else:
        st.warning("Please enter some text to classify.")
