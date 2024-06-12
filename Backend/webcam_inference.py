import cv2
import requests
import numpy as np
import time

# Define the API endpoint
url = 'http://127.0.0.1:5000/predict_single_frame'

def send_frame_to_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(url, files={'file': img_encoded.tostring()})
    return response.json()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_sent_time = time.time()
    prediction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the prediction result on the frame
        if prediction:
            cv2.putText(frame, f"Prediction: {prediction['prediction']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Webcam', frame)

        # Send one frame every second
        current_time = time.time()
        if current_time - last_sent_time >= 1:
            last_sent_time = current_time
            prediction = send_frame_to_api(frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
