import streamlit as st
import os
import pickle
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Medical Symptom Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #1E88E5;
    }
    .confidence-high {
        color: #388E3C;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #FFECB3;
        border-left: 5px solid #FFC107;
        padding: 10px;
        margin-top: 20px;
        border-radius: 5px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #1565C0;
    }
    .clear-button button {
        background-color: #757575;
        color: white;
    }
    .clear-button button:hover {
        background-color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except:
        nltk.download('punkt_tab')

download_nltk_resources()

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load model and related artifacts
@st.cache_resource
def load_model():
    """Load the trained model and related artifacts"""
    try:
        # Load classifier
        classifier = joblib.load('disease_classifier.joblib')
        
        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        # Load MultiLabelBinarizer
        mlb = joblib.load('multilabel_binarizer.joblib')
        
        # Load disease_to_symptoms dictionary
        with open('disease_to_symptoms.pkl', 'rb') as f:
            disease_to_symptoms = pickle.load(f)
        
        # Load symptom_to_diseases dictionary
        with open('symptom_to_diseases.pkl', 'rb') as f:
            symptom_to_diseases = pickle.load(f)
            
        return classifier, tfidf_vectorizer, mlb, disease_to_symptoms, symptom_to_diseases, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, False

# Function to predict diseases based on symptoms
def predict_disease(symptoms_description, classifier, tfidf_vectorizer, mlb, top_n=5):
    """Predict disease based on symptom description"""
    # Preprocess the input
    processed_input = preprocess_text(symptoms_description)
    
    # Vectorize the input
    input_tfidf = tfidf_vectorizer.transform([processed_input])
    
    # Get prediction scores
    prediction_scores = classifier.decision_function(input_tfidf)
    
    # Get top N diseases
    top_diseases_idx = np.argsort(prediction_scores[0])[::-1][:top_n]
    
    # Normalize scores to [0, 1] range
    max_score = np.max(prediction_scores)
    min_score = np.min(prediction_scores)
    normalized_scores = (prediction_scores[0] - min_score) / (max_score - min_score) if max_score > min_score else prediction_scores[0]
    
    results = []
    for idx in top_diseases_idx:
        disease = mlb.classes_[idx]
        score = normalized_scores[idx]
        if score > 0.1:  # Threshold to filter out unlikely matches
            results.append((disease, float(score)))
    
    return results

# Function to explain why a disease was predicted
def explain_prediction(disease, symptoms_description, disease_to_symptoms):
    """Explain why a disease was predicted based on matching symptoms"""
    if disease not in disease_to_symptoms:
        return []
    
    # Get known symptoms for the disease
    known_symptoms = disease_to_symptoms[disease]
    
    # Preprocess the input
    processed_input = preprocess_text(symptoms_description).split()
    
    # Find matching symptoms
    matching_symptoms = []
    for symptom in known_symptoms:
        processed_symptom = preprocess_text(symptom).split()
        # Check if any words from the symptom appear in the input
        if any(word in processed_input for word in processed_symptom):
            matching_symptoms.append(symptom)
    
    return matching_symptoms

# Function to get confidence level text and color
def get_confidence_class(score):
    if score >= 0.7:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

# Function to visualize symptom matches
def create_symptom_match_chart(matched_symptoms, all_symptoms):
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create data for visualization
    total_symptoms = len(all_symptoms)
    matched_count = len(matched_symptoms) 
    
    # Simple bar chart showing match ratio
    data = [matched_count, total_symptoms - matched_count]
    categories = ['Matched', 'Unmatched']
    colors = ['#1E88E5', '#E0E0E0']
    
    bars = ax.barh([0], [matched_count], height=0.4, color=colors[0], label='Matched')
    ax.barh([0], [total_symptoms - matched_count], height=0.4, left=[matched_count], color=colors[1], label='Unmatched')
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width/2, 
                bar.get_y() + bar.get_height()/2, 
                f"{matched_count} matched", 
                ha='center', 
                va='center',
                color='white',
                fontweight='bold')
    
    ax.text(matched_count + (total_symptoms - matched_count)/2, 
            0, 
            f"{total_symptoms - matched_count} other symptoms", 
            ha='center', 
            va='center',
            color='black')
    
    ax.set_yticks([])
    ax.set_xlabel('Number of Symptoms')
    ax.set_title(f'Symptom Match Analysis ({matched_count}/{total_symptoms} symptoms matched)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# Main application
def main():
    # Load models
    classifier, tfidf_vectorizer, mlb, disease_to_symptoms, symptom_to_diseases, models_loaded = load_model()
    
    # Header
    st.markdown("<h1 class='main-header'>Medical Symptom Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter your symptoms for disease prediction using AI</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/medicine-concept-illustration_114360-1192.jpg", width=280)
        
        st.markdown("### About")
        st.info(
            "This application uses Natural Language Processing and Machine Learning to predict "
            "potential diseases based on symptom descriptions. The model was trained on a comprehensive "
            "dataset mapping diseases to their associated symptoms."
        )
        
        # Display dataset stats
        if models_loaded:
            st.markdown("### Dataset Statistics")
            disease_count = len(disease_to_symptoms)
            symptom_count = len(set([symptom for symptoms in disease_to_symptoms.values() for symptom in symptoms]))
            
            col1, col2 = st.columns(2)
            col1.metric("Diseases", f"{disease_count}")
            col2.metric("Symptoms", f"{symptom_count}")
            
            # Display most common diseases with most symptoms
            st.markdown("### Top Diseases by Symptom Count")
            disease_symptom_counts = {disease: len(symptoms) for disease, symptoms in disease_to_symptoms.items()}
            top_diseases = sorted(disease_symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for disease, count in top_diseases:
                st.write(f"• {disease.title()}: {count} symptoms")
        
        # Example prompts
        st.markdown("### Example Symptom Descriptions")
        examples = [
            "I have a severe headache and fever",
            "Feeling dizzy with shortness of breath",
            "Chest pain, sweating, and arm numbness",
            "Itchy skin, rash, and difficulty breathing",
            "Joint pain, fatigue, and mild fever",
            "Stomach pain, nausea, and vomiting",
            "Persistent cough with yellow phlegm and chest tightness"
        ]
        
        example_button = st.selectbox("Try an example:", ["Select an example..."] + examples)
        
        st.markdown("### Disclaimer")
        st.warning(
            "This tool is for informational purposes only and not intended to provide medical advice. "
            "Always consult with qualified healthcare providers for medical advice, diagnosis, or treatment."
        )

    # Main area
    if not models_loaded:
        st.error("""
        Model files not found! Please run the training script first and ensure the following files are in the same directory:
        - disease_classifier.joblib
        - tfidf_vectorizer.joblib
        - multilabel_binarizer.joblib
        - disease_to_symptoms.pkl
        - symptom_to_diseases.pkl
        """)
        return
    
    # Set input text if example is selected
    if example_button != "Select an example...":
        user_input = example_button
    else:
        user_input = st.session_state.get("user_input", "")
    
    # User input area with custom styling
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Describe your symptoms:",
            value=user_input,
            height=120,
            key="symptom_input",
            placeholder="E.g., I've been experiencing headache and fever for the past three days..."
        )
        st.session_state.user_input = user_input
    
    with col2:
        st.write("")
        analyze_button = st.button("🔍 Analyze Symptoms", use_container_width=True)
        st.markdown("""<div class="clear-button">""", unsafe_allow_html=True)
        clear_button = st.button("🗑️ Clear", use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Process the input when button is clicked
    if clear_button:
        st.session_state.user_input = ""
        st.experimental_rerun()
    
    # Check for empty input
    if not user_input and analyze_button:
        st.warning("Please describe your symptoms before analysis.")
        return
    
    # Process the input and show results
    if user_input and (analyze_button or example_button != "Select an example..."):
        with st.spinner("Analyzing your symptoms..."):
            # Add a small delay for visual effect
            time.sleep(0.5)
            
            # Predict diseases
            predictions = predict_disease(user_input, classifier, tfidf_vectorizer, mlb)
            
            if not predictions:
                st.warning("😕 I couldn't identify any specific diseases based on these symptoms. Please provide more detailed symptoms.")
            else:
                st.success("✅ Analysis complete!")
                
                # Display results
                st.markdown("### Possible Conditions")
                
                for i, (disease, score) in enumerate(predictions):
                    # Format disease name
                    disease_name = disease.replace('_', ' ').title()
                    
                    # Get matching symptoms
                    matching_symptoms = explain_prediction(disease, user_input, disease_to_symptoms)
                    all_symptoms = disease_to_symptoms.get(disease, [])
                    
                    # Create confidence text with appropriate color class
                    confidence_class = get_confidence_class(score)
                    confidence_text = f"<span class='{confidence_class}'>{score:.0%}</span>"
                    
                    # Create expandable card for each disease
                    with st.expander(f"{disease_name} • Confidence: {confidence_text}", expanded=(i==0)):
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("#### Matching Symptoms:")
                            if matching_symptoms:
                                for symptom in matching_symptoms:
                                    formatted_symptom = symptom.replace('_', ' ')
                                    st.markdown(f"• {formatted_symptom}")
                            else:
                                st.info("No specific matching symptoms found, but the overall pattern of your description matches this condition.")
                            
                            st.markdown("#### Common Symptoms:")
                            symptom_list = [symptom.replace('_', ' ') for symptom in all_symptoms[:10]]
                            for symptom in symptom_list:
                                st.markdown(f"• {symptom}")
                            
                            if len(all_symptoms) > 10:
                                st.write(f"*...and {len(all_symptoms) - 10} more symptoms*")
                        
                        with col2:
                            # Create visualization of matched symptoms vs. total
                            if len(all_symptoms) > 0:
                                match_chart = create_symptom_match_chart(matching_symptoms, all_symptoms)
                                st.pyplot(match_chart)
                
                # Important disclaimer
                st.markdown("""
                <div class="disclaimer">
                    <strong>⚠️ IMPORTANT:</strong> This is not a professional medical diagnosis. 
                    The predictions are based on pattern matching and should not replace consultation 
                    with a qualified healthcare provider. Always seek professional medical advice for 
                    health concerns.
                </div>
                """, unsafe_allow_html=True)
                
                # Additional information - resources
                st.markdown("### Next Steps")
                st.markdown("""
                If you're concerned about these symptoms:
                - Consult with a healthcare provider
                - Keep track of your symptoms (duration, severity, triggers)
                - Don't self-medicate based solely on these predictions
                """)

if __name__ == "__main__":
    main()