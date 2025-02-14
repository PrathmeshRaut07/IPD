import collections
import pyaudio
import webrtcvad
import numpy as np
import os
import librosa
from google.cloud import speech
from textblob import TextBlob
from Functions.recorder import record_audio_stream
key_path=r"stoked-forest-447811-u4-e25fdb4e6e1c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

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

# Initialize Google Speech Client
speech_client = speech.SpeechClient()  # Ensure your credentials are set!

# -----------------------------
# Audio Recording and VAD Logic
# -----------------------------


# -----------------------------
# Google Speech-to-Text
# -----------------------------
def transcribe_audio(audio_data):
    """
    Sends the audio_data (raw PCM bytes) to Google Speech-to-Text and returns the transcript.
    """
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio)
    
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript
    return transcription

# -----------------------------
# Pitch Analysis Using Librosa
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

# -----------------------------
# Sentiment Analysis Using TextBlob
# -----------------------------
def analyze_sentiment(text):
    """
    Uses TextBlob to analyze the sentiment of the provided text.
    Returns a sentiment object with polarity (-1 to 1) and subjectivity.
    """
    blob = TextBlob(text)
    return blob.sentiment

# -----------------------------
# Processing Pipeline
# -----------------------------
def process_audio_segment(audio_data):
    print("\n--- Processing Audio Segment ---")
    
    # 1. Transcribe the audio segment
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)
    
    # 2. Analyze pitch from the audio segment
    pitch = analyze_pitch(audio_data)
    if pitch:
        print(f"Estimated Pitch: {pitch:.2f} Hz")
    else:
        print("Pitch analysis: Unable to estimate pitch.")
    
    # 3. Perform sentiment analysis on the transcript
    sentiment = analyze_sentiment(transcription)
    print("Sentiment Analysis:", sentiment)
    print("-" * 40)

# -----------------------------
# Main Loop
# -----------------------------
def main():
    for segment in record_audio_stream():
        process_audio_segment(segment)

if __name__ == "__main__":
    main()
