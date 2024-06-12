from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load your trained model
model = tf.keras.models.load_model('models/lrcn_model.h5')

def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the size used in training
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    return image

@app.route('/predict_batch_frames', methods=['POST'])
def predict_batch_frames():
    data = request.get_json()
    try:
        frames_data = data['frames']
        if len(frames_data) != 20:
            return jsonify({'error': 'Exactly 20 frames are required'}), 400

        frames = []
        for frame in frames_data:
            img_data = base64.b64decode(frame)
            image = Image.open(io.BytesIO(img_data))
            processed_image = preprocess_image(image)
            frames.append(processed_image)

        frames = np.array(frames)
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension

        logging.debug(f'Processed frames shape: {frames.shape}')
        
        prediction = model.predict(frames)
        logging.debug(f'Prediction: {prediction}')
        
        actions = {0: 'Climb', 1: 'Jump', 2: 'Normal'}
        action = actions[np.argmax(prediction)]
        
        return jsonify({'action': action})
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
