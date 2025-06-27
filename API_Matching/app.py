from flask import Flask, request, jsonify, send_file, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cache dataset features
dataset_features = None

def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1) if len(img_array.shape) == 2 else img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def get_dataset_features():
    global dataset_features
    if dataset_features is None:
        dataset_features = []
        dataset_files = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER) if allowed_file(f)]
        for file in dataset_files:
            try:
                features = extract_features(file)
                dataset_features.append((os.path.basename(file), features))
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return dataset_features

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:  # Changed from 'image' to 'file'
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract features from uploaded image
    uploaded_image_features = extract_features(file_path)

    # Compute similarity with dataset images
    dataset_features = get_dataset_features()
    similarities = []
    for filename, features in dataset_features:
        sim = cosine_similarity([uploaded_image_features], [features])[0][0]
        similarities.append((filename, sim))
    
    # Sort by similarity and get top 5
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_images = [os.path.basename(file) for file, _ in similarities[:5]]

    # Return URLs for similar images
    base_url = f'http://{request.host}/similar/'  # Dynamic base URL
    similar_image_urls = [f"{base_url}{filename}" for filename in similar_images]

    return jsonify({
        'message': 'Image processed successfully',
        'similar_images': similar_image_urls
    })

@app.route('/similar/<filename>', methods=['GET'])
def get_similar_image(filename):
    file_path = os.path.join(DATASET_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Image not found'}), 404
    return send_file(file_path, mimetype='image/jpeg')

@app.route('/', methods=['GET'])
def serve_frontend():
    return render_template('index.html')  # Use templates folder

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)