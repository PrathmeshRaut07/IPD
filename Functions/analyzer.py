import numpy as np
import librosa
import os
from google.cloud import speech
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pyaudio
FORMAT = pyaudio.paInt16      
CHANNELS = 1                  
RATE = 16000                 
FRAME_DURATION = 30         
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  
VAD_AGGRESSIVENESS = 3        # 0-3 (3 is most aggressive)
# -----------------------------
def analyze_pitch(audio_data):
    """
    Analyzes the audio_data for pitch using librosa.pyin.
    Returns the median pitch (in Hz) of the voiced frames.
    """
    # Convert raw bytes to numpy array (16-bit PCM)
    audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    # Normalize samples to range [-1, 1]
    audio_samples /= 32768.0

    try:
        # Use librosa's pyin for pitch estimation
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio_samples, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=RATE
        )
        # Filter out unvoiced frames (NaN values)
        if f0 is not None:
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                # Return the median pitch
                return np.median(voiced_f0)
    except Exception as e:
        print("Error in pitch analysis:", e)
    return None
def analyze_sentiment(text):
    """
    Uses TextBlob to analyze the sentiment of the provided text.
    Returns a sentiment object with polarity (-1 to 1) and subjectivity.
    """
    blob = TextBlob(text)
    return blob.sentiment
