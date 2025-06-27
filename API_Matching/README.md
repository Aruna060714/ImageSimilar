#SnapMatch
It is a web application that allows users to upload an image and find visually similar images from a dataset using deep learning-based image matching. Powered by Flask and TensorFlow, it leverages the ResNet50 model for feature extraction and cosine similarity for matching.

#Prerequisites
Python 3.12.3
Flask
TensorFlow
NumPy
Pillow (PIL)
scikit-learn

#Project Structure
app.py: Backend Flask application handling image uploads, feature extraction, and similarity matching.
templates/index.html: Frontend HTML file for the user interface.
uploads/: Temporary storage for uploaded images.
dataset/: Contains the dataset of images for matching.

#how it works
1.The application uses a pre-trained ResNet50 model to extract features from images.
2.Uploaded images are resized to 224x224 pixels and processed to extract feature vectors.
3.Cosine similarity is computed between the uploaded image's features and the dataset images' features.
4.The top 5 most similar images are returned.