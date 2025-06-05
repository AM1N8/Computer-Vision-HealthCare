from flask import Flask, render_template, request, jsonify, session
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
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK data already downloaded or download failed")

# Initialize global variables
lemmatizer = WordNetLemmatizer()
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Global variables for model and data
knn = None
intents = None
df = None
df_tr = None
vocab = None
disease = None
all_symp_col = None
all_symp = None
app_tag = None

# Load model and data
def load_model_and_data():
    global knn, intents, df, df_tr, vocab, disease, all_symp_col, all_symp, app_tag
    
    try:
        # Load KNN model
        knn = joblib.load('./model/knn.pkl')
        
        # Load intents
        with open('./Medical_dataset/intents_short.json', 'r') as f:
            intents = json.load(f)
        
        # Load datasets
        df = pd.read_csv('./Medical_dataset/tfidfsymptoms.csv')
        df_tr = pd.read_csv('./Medical_dataset/Training.csv')
        
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
                
        print("Model and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return False

# Preprocess sentence
def preprocess_sent(sent):
    try:
        t = nltk.word_tokenize(sent)
        return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t 
                        if (w not in set(stopwords.words('english')) and w.isalpha())])
    except:
        return sent.lower()

# Bag of words for preprocessed sentence
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Predict possible symptom in a sentence
def predictSym(sym, vocab, app_tag):
    sym = preprocess_sent(sym)
    bow = np.array(bag_of_words(sym, vocab))
    
    if df is None:
        return [], 0
        
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

# Input: patient symptoms / Output: OHV DataFrame 
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)], dtype=np.float64)
    for sym in cl_sym:
        if sym in all_sym:
            l[0, all_sym.index(sym)] = 1
    
    df_result = pd.DataFrame(l, columns=all_sym)
    return np.ascontiguousarray(df_result.values, dtype=np.float64)

def contains(small, big):
    return all(i in big for i in small)

# Returns possible diseases 
def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis

# Input: Disease / Output: all symptoms
def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# Preprocess symptoms    
def clean_symp(sym):
    return (sym.replace('_', ' ')
              .replace('.1', '')
              .replace('(typhos)', '')
              .replace('yellowish', 'yellow')
              .replace('yellowing', 'yellow'))

def getDescription():
    global description_list
    try:
        with open('./Medical_dataset/symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    _description = {row[0]: row[1]}
                    description_list.update(_description)
    except FileNotFoundError:
        print("Warning: symptom_Description.csv not found")

def getSeverityDict():
    global severityDictionary
    try:
        with open('./Medical_dataset/symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                try:
                    if len(row) >= 2:
                        _diction = {row[0]: int(row[1])}
                        severityDictionary.update(_diction)
                except ValueError:
                    continue
    except FileNotFoundError:
        print("Warning: symptom_severity.csv not found")

def getprecautionDict():
    global precautionDictionary
    try:
        with open('./Medical_dataset/symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 5:
                    _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                    precautionDictionary.update(_prec)
    except FileNotFoundError:
        print("Warning: symptom_precaution.csv not found")

def calc_condition(exp, days):
    sum_severity = 0
    for item in exp:
        if item in severityDictionary:
            sum_severity += severityDictionary[item]
    
    if len(exp) > 0 and ((sum_severity * days) / len(exp)) > 13:
        return 1, "You should take the consultation from doctor."
    else:
        return 0, "It might not be that bad but you should take precautions."

# Initialize chat state
def init_chat_state():
    return {
        'step': 'name',
        'name': '',
        'symptoms': [],
        'current_symptom_options': [],
        'current_symptom_index': 0,
        'possible_diseases': [],
        'additional_symptoms_asked': [],
        'current_disease_index': 0,
        'current_disease_symptoms': [],
        'current_disease_symptom_index': 0,
        'awaiting_days': False,
        'final_diagnosis': None
    }

@app.route('/')
def index():
    if 'chat_state' not in session:
        session['chat_state'] = init_chat_state()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'chat_state' not in session:
        session['chat_state'] = init_chat_state()
    
    user_message = request.json.get('message', '').strip()
    chat_state = session['chat_state']
    
    try:
        response = process_chat_message(user_message, chat_state)
        session['chat_state'] = chat_state
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'An error occurred: {str(e)}. Please try again or restart the conversation.'})

def process_chat_message(user_message, chat_state):
    if chat_state['step'] == 'name':
        if not user_message:
            return "Please tell me your name so I can assist you better."
        chat_state['name'] = user_message
        chat_state['step'] = 'first_symptom'
        return f"Hello {user_message}! Can you describe your main symptom?"
    
    elif chat_state['step'] == 'first_symptom':
        return handle_symptom_identification(user_message, chat_state, 'first')
    
    elif chat_state['step'] == 'confirm_first_symptom':
        return handle_symptom_confirmation(user_message, chat_state, 'first')
    
    elif chat_state['step'] == 'second_symptom':
        return handle_symptom_identification(user_message, chat_state, 'second')
    
    elif chat_state['step'] == 'confirm_second_symptom':
        return handle_symptom_confirmation(user_message, chat_state, 'second')
    
    elif chat_state['step'] == 'additional_symptoms':
        return handle_additional_symptoms(user_message, chat_state)
    
    elif chat_state['step'] == 'days_question':
        return handle_days_input(user_message, chat_state)
    
    elif chat_state['step'] == 'continue_question':
        return handle_continue_question(user_message, chat_state)
    
    else:
        return "I'm not sure how to help with that. Would you like to start over?"

def handle_symptom_identification(user_message, chat_state, symptom_type):
    psym, find = predictSym(user_message, vocab, app_tag)
    
    if find == 1:
        chat_state['symptoms'].append(psym)
        if symptom_type == 'first':
            chat_state['step'] = 'second_symptom'
            return f"I understand you have {psym.replace('_', ' ')}. Is there any other symptom you're experiencing?"
        else:
            return proceed_to_disease_analysis(chat_state)
    else:
        if len(psym) > 0:
            chat_state['current_symptom_options'] = psym
            chat_state['current_symptom_index'] = 0
            chat_state['step'] = f'confirm_{symptom_type}_symptom'
            return f"Do you experience {psym[0].replace('_', ' ')}? (yes/no)"
        else:
            return "I couldn't identify the symptom clearly. Could you describe it differently?"

def handle_symptom_confirmation(user_message, chat_state, symptom_type):
    response = user_message.lower().strip()
    
    if response == 'yes':
        symptom = chat_state['current_symptom_options'][chat_state['current_symptom_index']]
        chat_state['symptoms'].append(symptom)
        
        if symptom_type == 'first':
            chat_state['step'] = 'second_symptom'
            return f"Great! I've noted {symptom.replace('_', ' ')}. Is there any other symptom you're experiencing?"
        else:
            return proceed_to_disease_analysis(chat_state)
    
    elif response == 'no':
        chat_state['current_symptom_index'] += 1
        
        if chat_state['current_symptom_index'] < len(chat_state['current_symptom_options']):
            next_symptom = chat_state['current_symptom_options'][chat_state['current_symptom_index']]
            return f"Do you experience {next_symptom.replace('_', ' ')}? (yes/no)"
        else:
            return f"I couldn't identify the {symptom_type} symptom clearly. Could you describe it differently?"
    
    else:
        return "Please answer with 'yes' or 'no'."

def proceed_to_disease_analysis(chat_state):
    diseases = possible_diseases(chat_state['symptoms'])
    chat_state['possible_diseases'] = diseases
    chat_state['current_disease_index'] = 0
    chat_state['additional_symptoms_asked'] = []
    
    if len(diseases) == 0:
        return make_final_prediction(chat_state)
    
    return ask_additional_symptoms(chat_state)

def ask_additional_symptoms(chat_state):
    diseases = chat_state['possible_diseases']
    
    if chat_state['current_disease_index'] >= len(diseases):
        return make_final_prediction(chat_state)
    
    disease = diseases[chat_state['current_disease_index']]
    disease_symptoms = symVONdisease(df_tr, disease)
    
    # Find symptoms not already identified
    new_symptoms = [sym for sym in disease_symptoms 
                   if sym not in chat_state['symptoms'] 
                   and sym not in chat_state['additional_symptoms_asked']]
    
    if len(new_symptoms) > 0:
        chat_state['current_disease_symptoms'] = new_symptoms
        chat_state['current_disease_symptom_index'] = 0
        chat_state['step'] = 'additional_symptoms'
        
        symptom_name = clean_symp(new_symptoms[0])
        chat_state['additional_symptoms_asked'].append(new_symptoms[0])
        return f"Are you experiencing {symptom_name}? (yes/no)"
    else:
        chat_state['current_disease_index'] += 1
        return ask_additional_symptoms(chat_state)

def handle_additional_symptoms(user_message, chat_state):
    response = user_message.lower().strip()
    
    if response == 'yes':
        current_symptom = chat_state['current_disease_symptoms'][chat_state['current_disease_symptom_index']]
        chat_state['symptoms'].append(current_symptom)
        
        # Check if we now have a single possible disease
        diseases = possible_diseases(chat_state['symptoms'])
        if len(diseases) == 1:
            return make_final_prediction(chat_state)
    
    elif response != 'no':
        return "Please answer with 'yes' or 'no'."
    
    # Move to next symptom or disease
    chat_state['current_disease_symptom_index'] += 1
    
    if chat_state['current_disease_symptom_index'] < len(chat_state['current_disease_symptoms']):
        next_symptom = chat_state['current_disease_symptoms'][chat_state['current_disease_symptom_index']]
        symptom_name = clean_symp(next_symptom)
        chat_state['additional_symptoms_asked'].append(next_symptom)
        return f"Are you experiencing {symptom_name}? (yes/no)"
    else:
        chat_state['current_disease_index'] += 1
        return ask_additional_symptoms(chat_state)

def make_final_prediction(chat_state):
    try:
        ohv_result = OHV(chat_state['symptoms'], all_symp_col)
        prediction = knn.predict(ohv_result)
        predicted_disease = prediction[0]
        
        chat_state['final_diagnosis'] = predicted_disease
        chat_state['step'] = 'days_question'
        
        response = f"Based on your symptoms, you may have: **{predicted_disease}**\n\n"
        
        if predicted_disease in description_list:
            response += f"**Description:** {description_list[predicted_disease]}\n\n"
        
        response += "How many days have you been experiencing these symptoms?"
        
        return response
        
    except Exception as e:
        return f"I encountered an error making the prediction: {str(e)}. Please try describing your symptoms again."

def handle_days_input(user_message, chat_state):
    try:
        days = int(user_message.strip())
        severity_level, severity_message = calc_condition(chat_state['symptoms'], days)
        
        response = f"{severity_message}\n\n"
        
        if severity_level == 0:  # Not severe
            predicted_disease = chat_state['final_diagnosis']
            if predicted_disease in precautionDictionary:
                response += "**Recommended precautions:**\n"
                for precaution in precautionDictionary[predicted_disease]:
                    if precaution.strip():
                        response += f"• {precaution}\n"
            else:
                response += "No specific precautions available in our database."
        
        response += "\n\nDo you need another medical consultation? (yes/no)"
        chat_state['step'] = 'continue_question'
        
        return response
        
    except ValueError:
        return "Please enter a valid number of days."

def handle_continue_question(user_message, chat_state):
    response = user_message.lower().strip()
    
    if response == 'yes':
        # Reset chat state for new consultation
        session['chat_state'] = init_chat_state()
        return "Let's start a new consultation. What's your name?"
    else:
        # End conversation
        session['chat_state'] = init_chat_state()
        return "Thank you for using our medical consultation service! Take care and stay healthy. 🏥"

# Initialize the application
def initialize_app():
    if not load_model_and_data():
        print("Failed to load necessary data. Some features may not work.")
        return False
    
    getSeverityDict()
    getprecautionDict()
    getDescription()
    return True

@app.route('/reset')
def reset_chat():
    session['chat_state'] = init_chat_state()
    return jsonify({'status': 'Chat reset successfully'})

if __name__ == '__main__':
    if initialize_app():
        print("Medical Chatbot is ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize the application.")