import React, { useState, useRef, useEffect } from 'react';

const RecordedVideo = () => {
    const [videoUrl, setVideoUrl] = useState(null);
    const [action, setAction] = useState('Not Classified Yet');
    const [error, setError] = useState('');
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const framesBuffer = useRef([]);
    const intervalRef = useRef(null);
    const buzzerSound = useRef(null);

    useEffect(() => {
        buzzerSound.current = new Audio();
        buzzerSound.current.src = '/assets/alert_sound.weba'; // Ensure this path is correct
        buzzerSound.current.type = 'audio/webm'; // Specify the type if known
    }, []);

    const stopBuzzer = () => {
        if (buzzerSound.current) {
            buzzerSound.current.pause();
            buzzerSound.current.currentTime = 0;
        }
    };

    const handleChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                // Stop the current video processing
                stopFrameExtraction();
                stopBuzzer(); // Stop the buzzer if a new video is chosen
                setVideoUrl(event.target.result);
                // Clear previous results
                setAction('Not Classified Yet');
                setError('');
                framesBuffer.current = [];
            };
            reader.readAsDataURL(file);
        } else {
            setVideoUrl(null);
        }
    };

    const frameToBase64 = (blob) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                resolve(reader.result.split(',')[1]); // Get base64 string without the metadata
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    };

    const classifyFrames = async (frames) => {
        const payload = { frames };

        try {
            const response = await fetch('http://localhost:5000/predict_batch_frames', {
                method: 'POST',
                body: JSON.stringify(payload),
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorText = response.status === 413 ? 'File is too large. Please upload a smaller file.' : 'Failed to classify the video. Please check server.';
                throw new Error(errorText);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            setAction(result.action);

            // Buzzer logic
            if (result.action === 'Climb' || result.action === 'Jump') {
                buzzerSound.current.play();
            } else {
                stopBuzzer();
            }

        } catch (error) {
            console.error('Error:', error);
            setError(error.message || 'Unknown error occurred');
        }
    };

    const extractFrame = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (video && canvas) {
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(async (blob) => {
                framesBuffer.current.push(await frameToBase64(blob));
                if (framesBuffer.current.length >= 20) {
                    classifyFrames(framesBuffer.current);
                    framesBuffer.current = [];
                }
            }, 'image/jpeg');
        }
    };

    const startFrameExtraction = () => {
        intervalRef.current = setInterval(extractFrame, 150); // Adjusting interval to capture more frames
    };

    const stopFrameExtraction = () => {
        clearInterval(intervalRef.current);
    };

    const handleVideoPause = () => {
        stopFrameExtraction();
        stopBuzzer();
    };

    const handleVideoEnd = () => {
        stopFrameExtraction();
        stopBuzzer();
        setVideoUrl(null);
    };

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.addEventListener('play', startFrameExtraction);
            videoRef.current.addEventListener('pause', handleVideoPause);
            videoRef.current.addEventListener('ended', handleVideoEnd);
        }
        return () => {
            if (videoRef.current) {
                videoRef.current.removeEventListener('play', startFrameExtraction);
                videoRef.current.removeEventListener('pause', handleVideoPause);
                videoRef.current.removeEventListener('ended', handleVideoEnd);
            }
            stopFrameExtraction(); // Clear interval when component unmounts
        };
    }, [videoUrl]);

    const getBorderColor = () => {
        if (action === 'Not Classified Yet') {
            return 'black';
        } else if (action === 'Normal') {
            return 'green';
        } else {
            return 'red';
        }
    };

    return (
        <div className='recorded-video'>
            <div className='react-player-container' style={{ border: `10px solid ${getBorderColor()}`, padding: '10px' }}>
                {videoUrl && (
                    <video ref={videoRef} key={videoUrl} width="100%" height="100%" controls>
                        <source src={videoUrl} type="video/mp4" />
                        <source src={videoUrl} type="video/avi" />
                        Your browser does not support the video tag.
                    </video>
                )}
            </div>
            <canvas ref={canvasRef} style={{ display: 'none' }} width={640} height={360}></canvas>
            <div className="controls-container">
                <form className='recorded-video-form'>
                    <input className='recorded-video-input' type='file' accept="video/*" onChange={handleChange} name="file"/>
                </form>
            </div>
            <div className='video-results'>
                {error ? (
                    <div className="video-result error">
                        <span>Error: {error}</span>
                    </div>
                ) : (
                    <div className="video-result">
                        <span className="result-label">Action Predicted: </span>
                        <span className="video-action">{action}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default RecordedVideo;
