from flask import Flask, render_template, request, jsonify, session
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models
chest_yolo_model = None  # YOLO model for chest X-ray
bone_classifier_model = None
fracture_models = {'elbow': None, 'hand': None, 'shoulder': None}
models_loaded = {'chest': False, 'bone': False}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_yolo(image):
    """Preprocess image for YOLO model - no normalization needed"""
    image = image.convert('RGB')
    return image

def preprocess_image_for_keras(image, target_size=(224, 224)):
    """Preprocess image for Keras models"""
    image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def draw_yolo_predictions(image, results, confidence_threshold=0.25):
    """Draw YOLO predictions on image and return as base64"""
    try:
        # Create a copy of the original image
        img_copy = image.copy()
        
        # Convert PIL to numpy array
        img_array = np.array(img_copy)
        
        # Only convert to BGR if we're using OpenCV functions
        try:
            import cv2
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            use_opencv = True
        except ImportError:
            # Fallback to PIL drawing if OpenCV is not available
            use_opencv = False
            draw = ImageDraw.Draw(img_copy)
            try:
                # Try to load a font, fallback to default if not available
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        if len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()  # Box coordinates
                
                # Get class names safely
                class_names = {}
                if hasattr(chest_yolo_model, 'names'):
                    class_names = chest_yolo_model.names
                elif hasattr(chest_yolo_model, 'model') and hasattr(chest_yolo_model.model, 'names'):
                    class_names = chest_yolo_model.model.names
                
                # Draw boxes and labels
                for i, (box, conf, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
                    if conf > confidence_threshold:
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Prepare label
                        class_name = class_names.get(class_id, f'Detection_{class_id}')
                        label = f'{class_name}: {conf:.2f}'
                        
                        if use_opencv:
                            # Draw rectangle
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label background
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(img_cv, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                            
                            # Draw text
                            cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        else:
                            # Use PIL drawing as fallback
                            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                            draw.text((x1, y1 - 20), label, fill="green", font=font)
        
        if use_opencv:
            # Convert back to PIL
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        else:
            img_pil = img_copy
        
        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error drawing predictions: {e}")
        import traceback
        traceback.print_exc()
        
        # Return original image as base64 if drawing fails
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e2:
            print(f"Error creating fallback image: {e2}")
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """Return current model loading status"""
    return jsonify({
        'chest_loaded': models_loaded['chest'],
        'bone_loaded': models_loaded['bone']
    })

@app.route('/load_models', methods=['POST'])
def load_models():
    try:
        model_type = request.json.get('model_type')
        global chest_yolo_model, bone_classifier_model, fracture_models, models_loaded
        
        if model_type == 'chest':
            # Check if chest model is already loaded
            if models_loaded['chest']:
                return jsonify({'status': 'success', 'message': 'Chest X-ray YOLO model already loaded'})
            
            # Load chest X-ray YOLO model (.pt file)
            model_path = os.path.join('models', 'chest_xray_model.pt')
            if os.path.exists(model_path):
                try:
                    chest_yolo_model = YOLO(model_path)
                    models_loaded['chest'] = True
                    return jsonify({'status': 'success', 'message': 'Chest X-ray YOLO model loaded successfully'})
                except Exception as e:
                    return jsonify({'status': 'error', 'message': f'Error loading YOLO model: {str(e)}'})
            else:
                return jsonify({'status': 'error', 'message': f'Chest model file not found at {model_path}'})
                
        elif model_type == 'bone':
            # Check if bone models are already loaded
            if models_loaded['bone']:
                return jsonify({'status': 'success', 'message': 'Bone fracture models already loaded'})
            
            # Load bone fracture models (Keras .h5 files)
            classifier_path = os.path.join('models', 'bone_classifier.h5')
            elbow_path = os.path.join('models', 'elbow_fracture.h5')
            hand_path = os.path.join('models', 'hand_fracture.h5')
            shoulder_path = os.path.join('models', 'shoulder_fracture.h5')
            
            loaded_models = []
            
            if os.path.exists(classifier_path):
                bone_classifier_model = tf.keras.models.load_model(classifier_path)
                loaded_models.append('bone classifier')
            else:
                return jsonify({'status': 'error', 'message': f'Bone classifier not found at {classifier_path}'})
            
            if os.path.exists(elbow_path):
                fracture_models['elbow'] = tf.keras.models.load_model(elbow_path)
                loaded_models.append('elbow fracture')
            
            if os.path.exists(hand_path):
                fracture_models['hand'] = tf.keras.models.load_model(hand_path)
                loaded_models.append('hand fracture')
                
            if os.path.exists(shoulder_path):
                fracture_models['shoulder'] = tf.keras.models.load_model(shoulder_path)
                loaded_models.append('shoulder fracture')
            
            models_loaded['bone'] = True
            return jsonify({'status': 'success', 'message': f'Loaded: {", ".join(loaded_models)} models'})
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid model type'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error loading models: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})
        
        file = request.files['file']
        model_type = request.form.get('model_type')
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file format. Please upload PNG, JPG, or JPEG'})
        
        # Process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert original image to base64 for preview
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        original_image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        if model_type == 'chest':
            if not models_loaded['chest'] or chest_yolo_model is None:
                return jsonify({'status': 'error', 'message': 'Chest YOLO model not loaded. Please load the model first.'})
            
            # YOLO prediction
            results = chest_yolo_model.predict(image, verbose=False)
            
            # Draw predictions on image
            annotated_image_base64 = draw_yolo_predictions(image, results)
            
            # Extract predictions from YOLO results
            predictions_list = []
            
            if len(results) > 0:
                result = results[0]
                
                # Get class names from the model
                class_names = chest_yolo_model.names if hasattr(chest_yolo_model, 'names') else {}
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Object detection results
                    boxes = result.boxes
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # Group by class and get highest confidence for each
                    class_confidences = {}
                    for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                        class_name = class_names.get(class_id, f'Class_{class_id}')
                        if class_name not in class_confidences or conf > class_confidences[class_name]:
                            class_confidences[class_name] = float(conf)
                    
                    # Convert to list and sort
                    predictions_list = [(class_name, conf) for class_name, conf in class_confidences.items()]
                    predictions_list.sort(key=lambda x: x[1], reverse=True)
                
                else:
                    # Classification results or no detections
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Classification mode
                        probs = result.probs.data.cpu().numpy()
                        for i, prob in enumerate(probs):
                            class_name = class_names.get(i, f'Class_{i}')
                            predictions_list.append((class_name, float(prob)))
                        predictions_list.sort(key=lambda x: x[1], reverse=True)
                    else:
                        # No detections found
                        predictions_list = [('No findings detected', 0.0)]
            
            else:
                predictions_list = [('No results', 0.0)]
            
            return jsonify({
                'status': 'success',
                'model_type': 'chest',
                'predictions': predictions_list[:5],  # Top 5 predictions
                'original_image': f"data:image/png;base64,{original_image_base64}",
                'annotated_image': annotated_image_base64
            })
            
        elif model_type == 'bone':
            if not models_loaded['bone'] or bone_classifier_model is None:
                return jsonify({'status': 'error', 'message': 'Bone classifier not loaded. Please load the models first.'})
            
            # Preprocess for Keras models
            processed_image = preprocess_image_for_keras(image)
            
            # Step 1: Classify bone type
            bone_pred = bone_classifier_model.predict(processed_image, verbose=0)[0]
            bone_types = ['elbow', 'hand', 'shoulder']
            bone_type_idx = np.argmax(bone_pred)
            bone_type = bone_types[bone_type_idx] if bone_type_idx < len(bone_types) else 'unknown'
            bone_confidence = float(np.max(bone_pred))
            
            # Step 2: Check for fracture
            if fracture_models[bone_type] is None:
                return jsonify({'status': 'error', 'message': f'{bone_type.title()} fracture model not loaded'})
            
            fracture_pred = fracture_models[bone_type].predict(processed_image, verbose=0)[0]
            
            # Handle both binary and multi-class outputs
            if len(fracture_pred) == 1:
                # Binary classification (sigmoid output)
                fracture_prob = float(fracture_pred[0])
                is_fractured = bool(fracture_prob > 0.5)  # Convert to Python bool
            else:
                # Multi-class classification (softmax output)
                fracture_prob = float(fracture_pred[1])  # Assuming index 1 is 'fractured'
                is_fractured = bool(np.argmax(fracture_pred) == 1)  # Convert to Python bool
            
            return jsonify({
                'status': 'success',
                'model_type': 'bone',
                'bone_type': bone_type,
                'bone_confidence': bone_confidence,
                'is_fractured': is_fractured,
                'fracture_confidence': fracture_prob,
                'original_image': f"data:image/png;base64,{original_image_base64}"
            })
        
        else:
            return jsonify({'status': 'error', 'message': 'Invalid model type selected'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error during prediction: {str(e)}'})

@app.route('/test_model', methods=['POST'])
def test_model():
    """Test endpoint to check if YOLO model is working"""
    try:
        if not models_loaded['chest'] or chest_yolo_model is None:
            return jsonify({'status': 'error', 'message': 'Chest YOLO model not loaded'})
        
        # Test with a simple dummy image
        from PIL import Image
        import numpy as np
        
        # Create a test image (224x224 RGB)
        test_image = Image.new('RGB', (224, 224), color='gray')
        
        # Try to run prediction
        results = chest_yolo_model.predict(test_image, verbose=False)
        
        return jsonify({
            'status': 'success',
            'message': f'Model test successful. Got {len(results)} results.',
            'model_info': {
                'model_type': str(type(chest_yolo_model)),
                'has_names': hasattr(chest_yolo_model, 'names'),
                'names': chest_yolo_model.names if hasattr(chest_yolo_model, 'names') else 'No names attribute'
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': f'Model test failed: {str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chatbot endpoint with real-time status"""
    try:
        user_message = request.json.get('message', '').lower()
        
        # Check model status
        chest_status = "✅ loaded" if models_loaded['chest'] else "❌ not loaded"
        bone_status = "✅ loaded" if models_loaded['bone'] else "❌ not loaded"
        
        # Simple responses based on keywords
        if 'chest' in user_message:
            response = f"I can help analyze chest X-rays using YOLO detection. Model status: {chest_status}. Upload an image and select 'Chest X-ray' mode to see bounding boxes around detected findings."
        elif 'bone' in user_message or 'fracture' in user_message:
            response = f"I can detect bone fractures in elbow, hand, and shoulder X-rays. Model status: {bone_status}. Upload an image and select 'Bone Fracture' mode."
        elif 'model' in user_message or 'load' in user_message:
            response = f"Current model status:\n• Chest X-ray: {chest_status}\n• Bone fracture: {bone_status}\n\nUse the 'Load Models' buttons to load your trained models."
        elif 'help' in user_message:
            response = "I can help with:\n• Chest X-ray detection with visual annotations (YOLO)\n• Bone fracture detection (Elbow/Hand/Shoulder)\n• Loading and managing your AI models\n• Analyzing uploaded medical images with previews"
        elif 'status' in user_message:
            response = f"System Status:\n• Chest X-ray model: {chest_status}\n• Bone fracture models: {bone_status}\n• Ready to analyze images with visual feedback!"
        else:
            response = "Hi! I'm your medical imaging assistant. I can help with chest X-ray detection and bone fracture analysis. Upload images to see predictions with visual annotations!"
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Sorry, I encountered an error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)