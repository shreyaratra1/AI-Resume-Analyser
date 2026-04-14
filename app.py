import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('stopwords')

# Load models
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# UI Design
st.set_page_config(page_title="Resume Screening AI", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📄 AI Resume Analyzer</h1>", unsafe_allow_html=True)

st.write("Upload or paste your resume and get job role prediction instantly 🚀")

# Input
# Upload file
uploaded_file = st.file_uploader("📂 Upload Resume (txt)", type=["txt"])

# If file uploaded → read content
if uploaded_file is not None:
    resume_input = uploaded_file.read().decode("utf-8")
else:
    resume_input = st.text_area("📌 Paste Resume Text")

job_desc = st.text_area("📌 Paste Job Description (Optional for Matching)")

# Buttons
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict Role")

with col2:
    match_btn = st.button("📊 Match Score")

# Prediction
if predict_btn:
    if resume_input:
        cleaned = clean_text(resume_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        role = le.inverse_transform(prediction)

        st.success(f"✅ Predicted Job Role: {role[0]}")
    else:
        st.warning("⚠️ Please enter resume text")

# Matching
if match_btn:
    if resume_input and job_desc:
        resume_clean = clean_text(resume_input)
        job_clean = clean_text(job_desc)

        resume_vec = tfidf.transform([resume_clean])
        job_vec = tfidf.transform([job_clean])

        score = cosine_similarity(resume_vec, job_vec)[0][0]

        st.info(f"📊 Match Score: {round(score*100,2)}%")
    else:
        st.warning("⚠️ Please enter both resume and job description")
