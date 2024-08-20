from flask import Flask,request, jsonify
import pickle
from PIL import Image
import numpy as np
from flask import Flask, render_template

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image.reshape(1, 150, 150, 3)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file and preprocess it
    image = Image.open(file)
    processed_image = preprocess_image(image)

    # Perform prediction
    prediction = model.predict(processed_image)
    result = np.argmax(prediction, axis=1)[0]  # Assuming you're dealing with a classification problem

    # Return result
    if(int(result)==0):
        ans="Glioma Tumor"
    elif(int(result)==1):
        ans="Meningioma Tumor"
    elif(int(result)==2):
        ans="No Tumor"
    elif(int(result)==3):
        ans="Pituitary Tumor"

    return jsonify({'prediction': ans})

if __name__ == "__main__":
    app.run(debug=True)
