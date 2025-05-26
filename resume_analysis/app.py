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

# Set page config
st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Sidebar for file upload
st.sidebar.title("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Main content
st.title("Resume Analyzer and Job Role Predictor")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    comp_dt = pd.read_csv("company_data_set.csv")
    return df, comp_dt

df, comp_dt = load_data()

# Clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Category mapping
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

# Find companies
def find_companies(predicted_category_name_input, score_input):
    filtered_companies = comp_dt[(comp_dt['Category'] == predicted_category_name_input) & (comp_dt['Scale'] <= score_input)]
    company_names = filtered_companies['Company'].tolist()
    return company_names

def find_ineligible_companies(predicted_category_name_input, score_input):
    filtered_companies = comp_dt[(comp_dt['Category'] == predicted_category_name_input) & (comp_dt['Scale'] > score_input)]
    company_names = filtered_companies['Company'].tolist()
    return company_names

# Train and save models (this would normally be done separately)
@st.cache_resource
def get_models():
    # Preprocess data
    df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    le = LabelEncoder()
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(df['Resume'])
    requredTaxt = tfidf.transform(df['Resume'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)
    
    # Train classifier
    classifier = OneVsRestClassifier(SVC(probability=True))
    classifier.fit(X_train, y_train)
    
    return tfidf, classifier

tfidf, classifier = get_models()

# Main analysis function
def analyze_resume(resume_text):
    # Clean the input resume
    cleaned_resume = cleanResume(resume_text)
    
    # Transform the cleaned resume using the trained TfidfVectorizer
    input_features = tfidf.transform([cleaned_resume])
    
    # Make the prediction using the loaded classifier
    predicted_category_id = classifier.predict(input_features)[0]
    
    # Get the probabilities of each class
    probabilities = classifier.predict_proba(input_features)[0]
    
    # Find the probability of the predicted class
    predicted_category_probability = probabilities[predicted_category_id]
    
    # Convert the probability to a scale of 1 to 100
    strength = predicted_category_probability * 100
    
    # Map category ID to category name
    predicted_category_name = category_mapping.get(predicted_category_id, "Unknown")
    
    score = strength
    if score < 50:
        score += 45
    elif score > 50 and score < 75:
        score += 15
    
    return predicted_category_name, score, probabilities

# Display results
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
        