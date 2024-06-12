from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import logging


app = Flask(__name__)
CORS(app)

model = load_model('action_recognition_stateful_model.h5')  # Load your model

logging.basicConfig(level=logging.DEBUG)

def frames_to_array(frames):
    frame_list = []
    for frame in frames:
        # Convert the file storage object to a numpy array
        file_str = frame.read()
        np_arr = np.frombuffer(file_str, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Resize to uniform dimensions
        frame_list.append(img)
    return np.array(frame_list)

# Define the prediction function using the loaded model
def predict_action_loaded(frames, model):
    if len(frames) != 20:
        print("Error: Did not receive exactly 20 frames.")
        return None
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)
    predicted_class = np.argmax(prediction, axis=1)
    actions = {0: 'Climb', 1: 'Jump', 2: 'Normal'}
    return actions[predicted_class[0]]

@app.route('/action_predict_live', methods=['POST'])
def predict():
    frames = []
    for i in range(20):
        frame_key = f'frame_{i}'
        if frame_key in request.files:
            frames.append(request.files[frame_key])
        else:
            logging.error("Error processing frames")
            return jsonify({'error': f'Missing frame {i}'}), 400
    
    logging.info(f"Received {len(frames)} frames")
    frames_array = frames_to_array(frames)
    logging.debug(f"Frames array shape: {frames_array.shape}")
    prediction = predict_action_loaded(frames_array, model)
    if prediction is None:
        return jsonify({'error': 'Error processing frames'}), 400
    logging.info(f"Prediction result: {prediction}")

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
