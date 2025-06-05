# Medical Chatbot Flask Application

## Project Structure

```
medical_chatbot/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # HTML template
├── requirements.txt                # Python dependencies
├── Medical_dataset/                # Your dataset directory
│   ├── intents_short.json
│   ├── tfidfsymptoms.csv
│   ├── Training.csv
│   ├── symptom_Description.csv
│   ├── symptom_severity.csv
│   └── symptom_precaution.csv
└── model/
    └── knn.pkl                     # Your trained KNN model
```

## Installation and Setup

### 1. Create the directory structure:
```bash
mkdir medical_chatbot
cd medical_chatbot
mkdir templates
mkdir Medical_dataset
mkdir model
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Place your files:
- Save the Flask app code as `app.py`
- Save the HTML template as `templates/index.html`
- Copy your dataset files to the `Medical_dataset/` directory
- Copy your trained model to the `model/` directory

### 4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Features

### 🎯 **Interactive Web Interface**
- Modern, responsive design that works on desktop and mobile
- Real-time chat interface with typing indicators
- Clean, medical-themed UI with professional colors

### 🤖 **Intelligent Symptom Analysis**
- Progressive symptom identification using NLP
- Smart symptom confirmation with multiple options
- Disease prediction based on symptom combinations

### 📊 **Medical Intelligence**
- Severity assessment based on symptom duration
- Personalized precautions and recommendations
- Disease descriptions and explanations

### 🔄 **Session Management**
- Persistent chat state across interactions
- Ability to reset and start new consultations
- Multi-step conversation flow

### 🛡️ **Safety Features**
- Medical disclaimer prominently displayed
- Encourages professional medical consultation for serious cases
- Error handling and graceful degradation

## API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Process chat messages
- `GET /reset` - Reset chat session

## Customization

### Styling
The CSS in the HTML template can be customized to match your branding:
- Colors: Modify the gradient backgrounds and accent colors
- Fonts: Change the font family in the CSS
- Layout: Adjust container sizes and spacing

### Functionality
- Modify the chat flow in the `process_chat_message()` function
- Add new features like symptom history or user accounts
- Integrate with external medical APIs

### Data Sources
- Update dataset paths in the `load_model_and_data()` function
- Add new medical data sources
- Modify the symptom processing logic

## Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Security Considerations
- Change the Flask secret key in production
- Implement HTTPS in production
- Add rate limiting for API endpoints
- Validate and sanitize user inputs

## Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment
- Use a WSGI server like Gunicorn
- Configure reverse proxy with Nginx
- Set up SSL certificates
- Configure environment variables

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Troubleshooting

### Common Issues
1. **NLTK Download Errors**: Ensure internet connection for initial NLTK data download
2. **Model Loading Errors**: Check file paths and permissions
3. **CSV File Errors**: Verify CSV file format and encoding
4. **Memory Issues**: Monitor RAM usage with large datasets

### Performance Optimization
- Implement caching for repeated symptom lookups
- Optimize model loading time
- Add connection pooling for concurrent users
- Use async processing for heavy computations

## License and Disclaimer
This is a demonstration application. Always consult healthcare professionals for medical advice. The AI predictions are for educational purposes only.