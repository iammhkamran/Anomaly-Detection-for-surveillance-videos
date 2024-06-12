from flask import Flask, request, jsonify
import cv2
import os
import tempfile
import mediapipe
import numpy as np
from model import load_model_with_weights_LRCN
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 350 * 1024 * 1024  # Limit set to 50 MB
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder where uploaded videos will be stored
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Limit upload size up to 100 MB

CLASSES_LIST = ["JumpRope", "JumpingJack", "Punch", "climb", "climb_stairs", "jump", "jumpingTrampoline", "pjump"]
LRCN_model = load_model_with_weights_LRCN("models/LRCN.h5", CLASSES_LIST)
LSTM_Anomaly_Model = load_model("models/anomaly_new.h5")
actions = ['CLIMB', 'JUMP']
action_recognition_model = load_model('models/action_recognition_model2.h5')



################# HELPER FUNCTIONS############
def video_to_frames(video_path, max_frames=50):
    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) == max_frames:
                break
            frame = cv2.resize(frame, (128, 128))  # Resize to uniform dimensions
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)

def predict_action_loaded(video_path, action_recognition_model):
    frames = video_to_frames(video_path)
    if frames.shape[0] != 50:
        print("Error: Video does not have exactly 50 frames.")
        return None
    frames = np.expand_dims(frames, axis=0)
    prediction = action_recognition_model.predict(frames)
    predicted_class = np.argmax(prediction, axis=1)
    actions = {0: 'Climb', 1: 'Jump', 2: 'Normal'}
    return actions[predicted_class[0]]


def load_model(model_path):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=(50, 258)),
        Dropout(0.2),
        LSTM(256, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(256, return_sequences=False, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(model_path)

    return model


def lrcn_predict_single_action(video_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)

    frames_list = []
    
    predicted_class_name = ''

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        success, frame = video_reader.read() 
        video_reader.release()

        if not success:
            break
        IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        normalized_frame = resized_frame / 155
        
        frames_list.append(normalized_frame)


    if frames_list:  # Check if frames_list is not empty
        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]
        predicted_label = predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]
        if predicted_class_name == "JumpRope" or  predicted_class_name == "JumpingJack" or predicted_class_name == "jump" or predicted_class_name == "jumpingTrampoline" or predicted_class_name == "pjump" or predicted_class_name == "Punch" or predicted_class_name == "punch":
            predicted_class_name="Jump"
        else:
            predicted_class_name=predicted_class_name 
        confidence = float(predicted_labels_probabilities[predicted_label])
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {confidence}')
        return predicted_class_name, confidence  # Return both values
    else:
        return None, None  # Return None if no frames were processed

def lstm_predict_live(video_content, lstm_model):
    sequence = []
    cap = cv2.VideoCapture(video_content)

    mp_holistic = mediapipe.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        toskip = int(frame_count // 50)
        if toskip == 0:
            toskip = 1

        frame_num = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            frame_num += toskip

#            # rotate video right way up
#            (h, w) = frame.shape[:2]
#            rotpoint = (w // 2, h // 2)
#            rotmat = cv2.getRotationMatrix2D(rotpoint, 180, 1.0)
#            dim = (w, h)
#            intermediateFrame = cv2.warpAffine(frame, rotmat, dim)

            # cropping
            size = frame.shape
            finalFrame = frame[80:(size[0] - 180), 0:(size[1] - 20)]
            cv2.imwrite('fff.jpg',finalFrame)

            # keypoint prediction
            image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False  # Image is no longer writeable
            results = holistic.process(image)  # Make prediction
            image.flags.writeable = True  # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

            # extract and append keypoints
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                21 * 3)
            keypoints = np.concatenate([pose, lh, rh])
            sequence.append(keypoints)

            if len(sequence) == 50:
                break

    cap.release()
    cv2.destroyAllWindows()
    
    sequence = np.expand_dims(sequence, axis=0)[0]
    res = lstm_model.predict(np.expand_dims(sequence, axis=0))
    prediction_accuracy = np.max(res)
    print("Accuracy :",prediction_accuracy,res)
    if prediction_accuracy < 0.70:
        return 'Normal'
   
    return str(['CLIMB', 'JUMP'][np.argmax(res)])



@app.route('/classify', methods=['POST'])
def lrcn_classify_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400

    file = request.files['video']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        result, confidence = lrcn_predict_single_action(file_path, 20)
        if result is None or confidence is None:
            return jsonify({'error': 'Failed to process video'}), 500
        return jsonify({'action': result, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)  # Ensure file is removed after processing


@app.route('/predict_video', methods=['POST'])
def lstm_live():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    # Save video to a temporary file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_video.mp4")
    video_file.save(video_path)

    try:
        action_result = lstm_predict_live(video_path, LSTM_Anomaly_Model)
        return jsonify({'action': action_result})
    finally:
        # Clean up: remove the temporary file
        os.remove(video_path)
        os.rmdir(temp_dir)


@app.route('/action_predict_live', methods=['POST'])
def action_predict_live():
    video_file = request.files['video']  # Get the video file from POST request
    video_path = 'temp_video.mp4'
    video_file.save(video_path)  # Save the file temporarily
    
    prediction = predict_action_loaded(video_path, action_recognition_model)
    print(prediction)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
