import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pdfplumber
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

st.set_page_config(page_title="Resume Analyzer", layout="wide")

st.sidebar.title("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

st.title("Resume Analyzer and Job Role Predictor")

@st.cache_data
def load_data():
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    comp_dt = pd.read_csv("company_data_set.csv")
    return df, comp_dt

df, comp_dt = load_data()

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Analyst",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def find_companies(predicted_category_name_input, score_input):
    filtered_companies = comp_dt[(comp_dt['Category'] == predicted_category_name_input) & (comp_dt['Scale'] <= score_input)]
    company_names = filtered_companies['Company'].tolist()
    return company_names

def find_ineligible_companies(predicted_category_name_input, score_input):
    filtered_companies = comp_dt[(comp_dt['Category'] == predicted_category_name_input) & (comp_dt['Scale'] > score_input)]
    company_names = filtered_companies['Company'].tolist()
    return company_names

@st.cache_resource
def get_models():
    df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    le = LabelEncoder()
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(df['Resume'])
    requredTaxt = tfidf.transform(df['Resume'])
    
    X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)
    
    classifier = OneVsRestClassifier(SVC(probability=True))
    classifier.fit(X_train, y_train)
    
    return tfidf, classifier

tfidf, classifier = get_models()

def analyze_resume(resume_text):
    cleaned_resume = cleanResume(resume_text)
    input_features = tfidf.transform([cleaned_resume])
    predicted_category_id = classifier.predict(input_features)[0]
    probabilities = classifier.predict_proba(input_features)[0]
    predicted_category_probability = probabilities[predicted_category_id]
    strength = predicted_category_probability * 100
    predicted_category_name = category_mapping.get(predicted_category_id, "Unknown")
    score = strength
    if score < 50:
        score += 45
    elif score > 50 and score < 75:
        score += 15
    
    return predicted_category_name, score, probabilities

if uploaded_file is not None:
    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        predicted_category_name, score, probabilities = analyze_resume(resume_text)
        
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predicted Job Role")
            st.markdown(f"**{predicted_category_name}**")
            
            st.subheader("Resume Strength Score")
            st.markdown(f"**{score:.1f}/100**")
            
            st.subheader("Eligible Companies")
            matching_companies = find_companies(predicted_category_name, score)
            if matching_companies:
                for company in matching_companies:
                    st.write(f"- {company}")
            else:
                st.write("No eligible companies found.")
        