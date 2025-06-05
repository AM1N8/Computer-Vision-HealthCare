import streamlit as st
import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Page configuration
st.set_page_config(
    page_title="Medical Symptom Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border: 1px solid #FF9800;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E8;
        border: 1px solid #4CAF50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'start'
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'possible_symptoms' not in st.session_state:
    st.session_state.possible_symptoms = []
if 'symptom_index' not in st.session_state:
    st.session_state.symptom_index = 0

# Global variables
@st.cache_resource
def initialize_nltk():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return WordNetLemmatizer()
    except:
        st.error("Failed to download NLTK data")
        return None

lemmatizer = initialize_nltk()

@st.cache_data
def load_dictionaries():
    """Load severity, description, and precaution dictionaries"""
    severity_dict = {}
    description_dict = {}
    precaution_dict = {}
    
    try:
        # Load severity dictionary
        if os.path.exists('../Medical_dataset/symptom_severity.csv'):
            with open('../Medical_dataset/symptom_severity.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            severity_dict[row[0]] = int(row[1])
                        except ValueError:
                            continue
        
        # Load description dictionary
        if os.path.exists('../Medical_dataset/symptom_Description.csv'):
            with open('../Medical_dataset/symptom_Description.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        description_dict[row[0]] = row[1]
        
        # Load precaution dictionary
        if os.path.exists('../Medical_dataset/symptom_precaution.csv'):
            with open('../Medical_dataset/symptom_precaution.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 5:
                        precaution_dict[row[0]] = [row[1], row[2], row[3], row[4]]
        
    except Exception as e:
        st.error(f"Error loading dictionaries: {e}")
    
    return severity_dict, description_dict, precaution_dict

@st.cache_resource
def load_model_and_data():
    """Load machine learning model and datasets"""
    try:
        # Load KNN model
        knn = joblib.load('../model/knn.pkl')
        
        # Load intents
        with open('../Medical_dataset/intents_short.json', 'r') as f:
            intents = json.load(f)
        
        # Load datasets
        df = pd.read_csv('../Medical_dataset/tfidfsymptoms.csv')
        df_tr = pd.read_csv('../Medical_dataset/Training.csv')
        
        # Process data
        vocab = list(df.columns)
        disease = df_tr.iloc[:, -1].to_list()
        all_symp_col = list(df_tr.columns[:-1])
        all_symp = [clean_symp(sym) for sym in all_symp_col]
        
        # Build app_tag list
        app_tag = []
        for intent in intents['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                app_tag.append(tag)
        
        return knn, df, df_tr, vocab, disease, all_symp_col, all_symp, app_tag, True
        
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None, None, None, None, False

def preprocess_sent(sent):
    """Preprocess sentence for symptom prediction"""
    if not lemmatizer:
        return sent.lower()
    try:
        t = nltk.word_tokenize(sent)
        return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t 
                        if (w not in set(stopwords.words('english')) and w.isalpha())])
    except:
        return sent.lower()

def bag_of_words(tokenized_sentence, all_words):
    """Create bag of words vector"""
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def predict_symptom(sym, vocab, app_tag, df):
    """Predict possible symptom in a sentence"""
    sym = preprocess_sent(sym)
    bow = np.array(bag_of_words(sym, vocab))
    
    res = cosine_similarity(bow.reshape((1, -1)), df.values).reshape(-1)
    order = np.argsort(res)[::-1].tolist()
    possym = []
    
    for i in order:
        if i < len(app_tag):
            if app_tag[i].replace('_', ' ') in sym:
                return app_tag[i], 1
            if app_tag[i] not in possym and res[i] != 0:
                possym.append(app_tag[i])
    return possym, 0

def clean_symp(sym):
    """Clean symptom names"""
    return (sym.replace('_', ' ')
              .replace('.1', '')
              .replace('(typhos)', '')
              .replace('yellowish', 'yellow')
              .replace('yellowing', 'yellow'))

def ohv_encode(cl_sym, all_sym):
    """One Hot Vector encoding for symptoms"""
    l = np.zeros([1, len(all_sym)], dtype=np.float64)
    for sym in cl_sym:
        if sym in all_sym:
            l[0, all_sym.index(sym)] = 1
    return np.ascontiguousarray(l, dtype=np.float64)

def contains(small, big):
    """Check if all elements in small list are in big list"""
    return all(i in big for i in small)

def possible_diseases(symptom_list, df_tr, disease_list):
    """Get possible diseases based on symptoms"""
    poss_dis = []
    for dis in set(disease_list):
        disease_symptoms = get_disease_symptoms(df_tr, dis)
        if contains(symptom_list, disease_symptoms):
            poss_dis.append(dis)
    return poss_dis

def get_disease_symptoms(df, disease):
    """Get all symptoms for a specific disease"""
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

def calculate_condition(symptoms, days, severity_dict):
    """Calculate condition severity"""
    sum_severity = 0
    for item in symptoms:
        if item in severity_dict:
            sum_severity += severity_dict[item]
    
    if len(symptoms) > 0 and ((sum_severity * days) / len(symptoms)) > 13:
        return 1  # Serious condition
    else:
        return 0  # Mild condition

def add_message(role, content):
    """Add message to conversation history"""
    st.session_state.conversation_history.append({"role": role, "content": content})

def display_conversation():
    """Display conversation history"""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 **You:** {msg["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 **Medical Assistant:** {msg["content"]}</div>', 
                       unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Symptom Chatbot</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This chatbot is for informational purposes only and should not replace professional medical advice. 
        Always consult with a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    if not st.session_state.model_loaded:
        with st.spinner("Loading medical data and AI model..."):
            knn, df, df_tr, vocab, disease, all_symp_col, all_symp, app_tag, success = load_model_and_data()
            if success:
                st.session_state.knn = knn
                st.session_state.df = df
                st.session_state.df_tr = df_tr
                st.session_state.vocab = vocab
                st.session_state.disease = disease
                st.session_state.all_symp_col = all_symp_col
                st.session_state.all_symp = all_symp
                st.session_state.app_tag = app_tag
                st.session_state.model_loaded = True
                
                # Load dictionaries
                severity_dict, description_dict, precaution_dict = load_dictionaries()
                st.session_state.severity_dict = severity_dict
                st.session_state.description_dict = description_dict
                st.session_state.precaution_dict = precaution_dict
            else:
                st.error("Failed to load medical data. Please check if all required files are present.")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Session Info")
        if st.session_state.user_name:
            st.write(f"**Patient:** {st.session_state.user_name}")
        if st.session_state.symptoms:
            st.write("**Recorded Symptoms:**")
            for i, sym in enumerate(st.session_state.symptoms, 1):
                st.write(f"{i}. {clean_symp(sym)}")
        
        st.markdown("---")
        if st.button("🔄 Start New Consultation"):
            # Reset session state
            for key in ['conversation_history', 'current_step', 'user_name', 'symptoms', 
                       'possible_symptoms', 'symptom_index']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_step = 'start'
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Consultation")
        
        # Display conversation
        if st.session_state.conversation_history:
            display_conversation()
        
        # Chat logic based on current step
        if st.session_state.current_step == 'start':
            st.markdown("**🤖 Medical Assistant:** Hello! I'm here to help you identify potential health conditions based on your symptoms. May I have your name?")
            
            user_name = st.text_input("Enter your name:", key="name_input")
            if st.button("Continue") and user_name:
                st.session_state.user_name = user_name
                st.session_state.current_step = 'first_symptom'
                add_message("user", user_name)
                add_message("bot", f"Hello {user_name}! Can you describe your main symptom?")
                st.rerun()
        
        elif st.session_state.current_step == 'first_symptom':
            symptom1 = st.text_input("Describe your main symptom:", key="symptom1_input")
            if st.button("Submit First Symptom") and symptom1:
                add_message("user", symptom1)
                
                # Predict symptom
                psym1, find = predict_symptom(symptom1, st.session_state.vocab, 
                                            st.session_state.app_tag, st.session_state.df)
                
                if find == 1:
                    st.session_state.symptoms.append(psym1)
                    st.session_state.current_step = 'second_symptom'
                    add_message("bot", f"I understand you're experiencing {clean_symp(psym1)}. Is there any other symptom you're experiencing?")
                else:
                    st.session_state.possible_symptoms = psym1[:5]  # Limit to top 5
                    st.session_state.symptom_index = 0
                    st.session_state.current_step = 'confirm_first_symptom'
                    add_message("bot", "Let me help you identify your symptom more precisely.")
                st.rerun()
        
        elif st.session_state.current_step == 'confirm_first_symptom':
            if st.session_state.symptom_index < len(st.session_state.possible_symptoms):
                current_sym = st.session_state.possible_symptoms[st.session_state.symptom_index]
                st.markdown(f"**🤖 Medical Assistant:** Do you experience {clean_symp(current_sym)}?")
                
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes", key=f"yes_first_{st.session_state.symptom_index}"):
                        st.session_state.symptoms.append(current_sym)
                        st.session_state.current_step = 'second_symptom'
                        add_message("user", "Yes")
                        add_message("bot", "Is there any other symptom you're experiencing?")
                        st.rerun()
                
                with col_no:
                    if st.button("No", key=f"no_first_{st.session_state.symptom_index}"):
                        st.session_state.symptom_index += 1
                        add_message("user", "No")
                        st.rerun()
            else:
                st.error("Could not identify your first symptom. Please try describing it differently.")
                st.session_state.current_step = 'first_symptom'
        
        elif st.session_state.current_step == 'second_symptom':
            symptom2 = st.text_input("Describe any other symptom:", key="symptom2_input")
            if st.button("Submit Second Symptom") and symptom2:
                add_message("user", symptom2)
                
                # Predict second symptom
                psym2, find = predict_symptom(symptom2, st.session_state.vocab, 
                                            st.session_state.app_tag, st.session_state.df)
                
                if find == 1:
                    st.session_state.symptoms.append(psym2)
                    st.session_state.current_step = 'additional_symptoms'
                    add_message("bot", f"I understand you're also experiencing {clean_symp(psym2)}. Let me ask about some additional symptoms to get a better diagnosis.")
                else:
                    st.session_state.possible_symptoms = psym2[:5]
                    st.session_state.symptom_index = 0
                    st.session_state.current_step = 'confirm_second_symptom'
                    add_message("bot", "Let me help identify your second symptom.")
                st.rerun()
        
        elif st.session_state.current_step == 'confirm_second_symptom':
            if st.session_state.symptom_index < len(st.session_state.possible_symptoms):
                current_sym = st.session_state.possible_symptoms[st.session_state.symptom_index]
                st.markdown(f"**🤖 Medical Assistant:** Do you experience {clean_symp(current_sym)}?")
                
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes", key=f"yes_second_{st.session_state.symptom_index}"):
                        st.session_state.symptoms.append(current_sym)
                        st.session_state.current_step = 'additional_symptoms'
                        add_message("user", "Yes")
                        add_message("bot", "Let me ask about some additional symptoms to get a better diagnosis.")
                        st.rerun()
                
                with col_no:
                    if st.button("No", key=f"no_second_{st.session_state.symptom_index}"):
                        st.session_state.symptom_index += 1
                        add_message("user", "No")
                        st.rerun()
            else:
                st.session_state.current_step = 'additional_symptoms'
                add_message("bot", "Let me ask about some additional symptoms based on what you've told me.")
                st.rerun()
        
        elif st.session_state.current_step == 'additional_symptoms':
            # Get possible diseases and ask about additional symptoms
            diseases = possible_diseases(st.session_state.symptoms, st.session_state.df_tr, st.session_state.disease)
            
            if diseases:
                # Get additional symptoms to ask about
                additional_symptoms = []
                for dis in diseases:
                    disease_symptoms = get_disease_symptoms(st.session_state.df_tr, dis)
                    for sym in disease_symptoms:
                        if sym not in st.session_state.symptoms and sym not in additional_symptoms:
                            additional_symptoms.append(sym)
                
                if additional_symptoms and len(additional_symptoms) > 0:
                    if 'additional_symptom_index' not in st.session_state:
                        st.session_state.additional_symptom_index = 0
                    
                    if st.session_state.additional_symptom_index < len(additional_symptoms) and st.session_state.additional_symptom_index < 5:
                        current_sym = additional_symptoms[st.session_state.additional_symptom_index]
                        st.markdown(f"**🤖 Medical Assistant:** Are you also experiencing {clean_symp(current_sym)}?")
                        
                        col_yes, col_no, col_skip = st.columns(3)
                        with col_yes:
                            if st.button("Yes", key=f"yes_additional_{st.session_state.additional_symptom_index}"):
                                st.session_state.symptoms.append(current_sym)
                                st.session_state.additional_symptom_index += 1
                                add_message("user", "Yes")
                                
                                # Check if we can make a diagnosis
                                updated_diseases = possible_diseases(st.session_state.symptoms, st.session_state.df_tr, st.session_state.disease)
                                if len(updated_diseases) == 1:
                                    st.session_state.current_step = 'diagnosis'
                                st.rerun()
                        
                        with col_no:
                            if st.button("No", key=f"no_additional_{st.session_state.additional_symptom_index}"):
                                st.session_state.additional_symptom_index += 1
                                add_message("user", "No")
                                st.rerun()
                        
                        with col_skip:
                            if st.button("Skip to Diagnosis", key="skip_to_diagnosis"):
                                st.session_state.current_step = 'diagnosis'
                                add_message("user", "Skip to diagnosis")
                                st.rerun()
                    else:
                        st.session_state.current_step = 'diagnosis'
                        st.rerun()
                else:
                    st.session_state.current_step = 'diagnosis'
                    st.rerun()
            else:
                st.session_state.current_step = 'diagnosis'
                st.rerun()
        
        elif st.session_state.current_step == 'diagnosis':
            # Make prediction
            if len(st.session_state.symptoms) >= 2:
                ohv_result = ohv_encode(st.session_state.symptoms, st.session_state.all_symp_col)
                prediction = st.session_state.knn.predict(ohv_result)
                predicted_disease = prediction[0]
                
                st.markdown(f"**🤖 Medical Assistant:** Based on your symptoms, you may have: **{predicted_disease}**")
                add_message("bot", f"Based on your symptoms, you may have: {predicted_disease}")
                
                # Show description
                if predicted_disease in st.session_state.description_dict:
                    description = st.session_state.description_dict[predicted_disease]
                    st.markdown(f"**Description:** {description}")
                    add_message("bot", f"Description: {description}")
                
                # Ask for duration
                st.session_state.current_step = 'duration'
                st.rerun()
            else:
                st.error("Need at least 2 symptoms for diagnosis. Please restart the consultation.")
        
        elif st.session_state.current_step == 'duration':
            days = st.number_input("How many days have you been experiencing these symptoms?", 
                                 min_value=1, max_value=365, value=1, key="days_input")
            
            if st.button("Get Recommendations"):
                add_message("user", f"{days} days")
                
                # Calculate condition severity
                condition = calculate_condition(st.session_state.symptoms, days, st.session_state.severity_dict)
                
                if condition == 1:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Recommendation:</strong> You should consult with a doctor as soon as possible.
                    </div>
                    """, unsafe_allow_html=True)
                    add_message("bot", "Recommendation: You should consult with a doctor as soon as possible.")
                else:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Recommendation:</strong> It might not be serious, but please take the following precautions:
                    </div>
                    """, unsafe_allow_html=True)
                    add_message("bot", "It might not be serious, but please take precautions.")
                    
                    # Show precautions
                    ohv_result = ohv_encode(st.session_state.symptoms, st.session_state.all_symp_col)
                    prediction = st.session_state.knn.predict(ohv_result)
                    predicted_disease = prediction[0]
                    
                    if predicted_disease in st.session_state.precaution_dict:
                        st.markdown("**Precautions:**")
                        precautions = st.session_state.precaution_dict[predicted_disease]
                        for i, precaution in enumerate(precautions, 1):
                            if precaution.strip():
                                st.markdown(f"{i}. {precaution}")
                        
                        precaution_text = "Precautions: " + "; ".join([p for p in precautions if p.strip()])
                        add_message("bot", precaution_text)
                
                st.session_state.current_step = 'complete'
                st.rerun()
        
        elif st.session_state.current_step == 'complete':
            st.markdown("**🤖 Medical Assistant:** Consultation completed! Would you like to start a new consultation?")
            
            if st.button("New Consultation"):
                # Reset for new consultation
                for key in ['conversation_history', 'current_step', 'user_name', 'symptoms', 
                           'possible_symptoms', 'symptom_index', 'additional_symptom_index']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.current_step = 'start'
                st.rerun()
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Enter your name** to start the consultation
        2. **Describe your main symptom** in your own words
        3. **Provide additional symptoms** when asked
        4. **Answer questions** about related symptoms
        5. **Get a diagnosis** based on AI analysis
        6. **Receive recommendations** and precautions
        """)
        
        st.markdown("---")
        st.subheader("🔒 Privacy")
        st.markdown("""
        - No data is stored permanently
        - Session resets when you refresh
        - All processing is local
        """)

if __name__ == "__main__":
    main()