import React, { useRef, useState, useEffect } from 'react';
import Webcam from "react-webcam";

const LiveClassification = () => {
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [recordedChunks, setRecordedChunks] = useState([]);
    const [action, setAction] = useState('Not Classified Yet');
    const [recording, setRecording] = useState(false);
    const isProcessingAllowed = useRef(true);  // Control flag for processing
    const buzzerSound = useRef(null);


    useEffect(() => {
        buzzerSound.current = new Audio();
        buzzerSound.current.src = '/assets/alert_sound.weba'; // Ensure this path is correct
        buzzerSound.current.type = 'audio/webm'; // Specify the type if known
    }, []);

    const videoConstraints = {

        width: { min: 1200 },
        height: { min: 720 },
        aspectRatio: 16.4666666667,
        facingMode: "user"
    };

    const handleDataAvailable = ({ data }) => {
        if (data.size > 0 && isProcessingAllowed.current) {
            sendVideoToServer(data);
            // saveVideoFile(data);
        }
    };
    const saveVideoFile = (videoBlob) => {
        const url = window.URL.createObjectURL(videoBlob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'recorded_video.webm';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    };

    const sendVideoToServer = async (videoBlob) => {
        if (!isProcessingAllowed.current) return;

        const formData = new FormData();
        formData.append('video', videoBlob, 'video.webm');
        setAction('Predicting...');

        try {
            const response = await fetch('http://127.0.0.1:5000/action_predict_live', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) throw new Error('Network response was not ok.');
            const data = await response.json();
            if (isProcessingAllowed.current) {
                setAction(data.prediction);

              
                if (data.prediction === 'Climb' || data.prediction === 'Jump') {
                    buzzerSound.current.play(); // Play the buzzer sound for Climb or Jump
                }
              
                else {
                    buzzerSound.current.pause(); // Pause the buzzer sound if the action is 'Normal'
                    buzzerSound.current.currentTime = 0; // Optionally reset the audio position to the start
                }
                startRecordingProcess();

               
            }
        } catch (error) {
            console.error('Error sending video to server:', error);
            setAction('Error processing video. Check Server');
            if (recording && isProcessingAllowed.current) {
                startRecordingProcess();
            }
        }
    };

    const startRecordingProcess = () => {
        if (!isProcessingAllowed.current) return;

        mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
            mimeType: "video/webm"
        });
        mediaRecorderRef.current.addEventListener("dataavailable", handleDataAvailable);
        mediaRecorderRef.current.start();
        
        // Automatically stop recording after 5 seconds
        setTimeout(() => {
            mediaRecorderRef.current.stop();
        }, 7000);
    };

    const startRecording = () => {
        isProcessingAllowed.current = true;  // Allow processing
        setRecording(true);
        startRecordingProcess();
    };

    const stopRecording = () => {
        isProcessingAllowed.current = false;  // Stop all processing
        if (recording) {
            setRecording(false);
            if (mediaRecorderRef.current) {
                mediaRecorderRef.current.stop();
                mediaRecorderRef.current.removeEventListener("dataavailable", handleDataAvailable);
            }
        }
        setAction("Not Classifying Currently")
    };

    useEffect(() => {
        return () => {
            stopRecording();
        };
    }, []);

    return (
        <div className="live-classification">
            <Webcam
                audio={false}
                ref={webcamRef}
                videoConstraints={videoConstraints}
            />
            <div className="controls-container">
                <button className="classify-btn" onClick={startRecording} disabled={recording}>Start Processing</button>
                <button className="classify-btn" onClick={stopRecording} disabled={!recording}>Stop Processing</button>
            </div>
            <div className='video-results'>
                <div className="video-result">
                    <span className="result-label">Action Predicted: </span>
                    <span className="video-action">{action}</span>
                </div>
            </div>
        </div>
    );
};

export default LiveClassification;
