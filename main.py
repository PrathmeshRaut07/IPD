import collections
import pyaudio
import webrtcvad
import numpy as np
import os
import librosa
import cv2
import mediapipe as mp
from Functions.recorder import record_audio_stream
from Functions.saver import process_audio_segment

# Set Google credentials
key_path = r"stoked-forest-447811-u4-e25fdb4e6e1c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Audio configuration
FORMAT = pyaudio.paInt16      
CHANNELS = 1                  
RATE = 16000                 
FRAME_DURATION = 30         
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  
VAD_AGGRESSIVENESS = 3        # 0-3 (3 is most aggressive)
NUM_PADDING_FRAMES = int(300 / FRAME_DURATION)

# -----------------------------
# Initialize Audio and VAD
# -----------------------------
p = pyaudio.PyAudio()
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# -----------------------------
# Initialize Video Capture and MediaPipe Face Mesh for Posture Analysis
# -----------------------------
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def analyze_face_posture(frame, face_mesh):
    """
    Analyze face posture using MediaPipe Face Mesh.
    Returns a dictionary with metrics: whether a face was detected,
    the computed tilt angle, and a posture description.
    """
    # Convert the frame from BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    posture_metrics = {}
    if results.multi_face_landmarks:
        # Process the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Use two eye landmarks to estimate head tilt.
        # Landmark 33: left eye outer corner, Landmark 263: right eye outer corner.
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
        
        # Compute the angle (in degrees) of the line connecting the eyes relative to horizontal.
        dx = right_eye_coords[0] - left_eye_coords[0]
        dy = right_eye_coords[1] - left_eye_coords[1]
        import math
        angle = math.degrees(math.atan2(dy, dx))
        
        posture_metrics["face_detected"] = True
        posture_metrics["tilt_angle"] = angle
        
        # Determine posture description based on the tilt angle.
        if abs(angle) < 5:
            posture_metrics["posture"] = "Straight"
        elif angle > 5:
            posture_metrics["posture"] = "Tilted Right"
        else:
            posture_metrics["posture"] = "Tilted Left"
    else:
        posture_metrics["face_detected"] = False
        posture_metrics["posture"] = "No face detected"
        posture_metrics["tilt_angle"] = None
    
    return posture_metrics

def main():
    print("Starting audio and face analysis...")
    try:
        while True:
            # Get an audio segment from the microphone
            audio_segment = next(record_audio_stream())
            # Process the audio segment (transcription, analysis, RL updates, etc.)
            process_audio_segment(audio_segment)
            
            # Capture one frame from the webcam for face posture analysis.
            ret, frame = cap.read()
            if ret:
                posture_results = analyze_face_posture(frame, face_mesh)
                print("Face Posture Analysis:", posture_results)
            else:
                print("Warning: Could not capture video frame.")
            
    except KeyboardInterrupt:
        print("Stopping analysis...")
    finally:
        cap.release()
        p.terminate()
        face_mesh.close()

if __name__ == "__main__":
    main()
