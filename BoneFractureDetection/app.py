import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import datetime

from predictions import predict

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return encoded_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Convert image to base64 for preview
            image_data = image_to_base64(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'image_data': f"data:image/jpeg;base64,{image_data}"
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

@app.route('/predict', methods=['POST'])
def predict_fracture():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Get bone type prediction
        bone_type_result = predict(filepath, "Parts")
        
        # Get fracture prediction
        fracture_result = predict(filepath)
        final_result = predict(filepath, fracture_result)
        
        # Determine result status and color
        if final_result.lower() == 'fractured':
            result_status = 'Fractured'
            result_color = 'red'
        else:
            result_status = 'Normal'
            result_color = 'green'
        
        return jsonify({
            'success': True,
            'bone_type': bone_type_result,
            'fracture_status': result_status,
            'result_color': result_color,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.get_json()
    filename = data.get('filename')
    bone_type = data.get('bone_type')
    fracture_status = data.get('fracture_status')
    
    if not all([filename, bone_type, fracture_status]):
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        # Load original image
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = Image.open(original_path)
        
        # Create result filename
        result_filename = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Save the image (you could add text overlay here if needed)
        image.save(result_path)
        
        # Create a text file with results
        text_filename = result_filename.replace('.png', '.txt')
        text_path = os.path.join(app.config['RESULTS_FOLDER'], text_filename)
        
        with open(text_path, 'w') as f:
            f.write(f"Bone Fracture Detection Results\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original File: {filename}\n")
            f.write(f"Bone Type: {bone_type}\n")
            f.write(f"Fracture Status: {fracture_status}\n")
        
        return jsonify({
            'success': True,
            'message': 'Results saved successfully!',
            'result_filename': result_filename
        })
    except Exception as e:
        return jsonify({'error': f'Error saving results: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)