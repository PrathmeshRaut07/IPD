import numpy as np
import librosa
import os
from google.cloud import speech
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
key_path=r"stoked-forest-447811-u4-e25fdb4e6e1c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
def analyze_audio(filepath, rate):
    print(f"\nAnalyzing audio: {filepath}")

    # Load audio
    y, sr_rate = librosa.load(filepath, sr=rate)

    # --- Pitch Analysis ---
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = np.nanmean(f0) if f0 is not None and np.count_nonzero(~np.isnan(f0)) > 0 else None

    # --- Speech-to-Text using Google Cloud Speech ---
    client = speech.SpeechClient()

    # Read the audio file
    with open(filepath, "rb") as audio_file:
        audio_content = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code="en-US"
    )

    try:
        response = client.recognize(config=config, audio=audio)
        text = response.results[0].alternatives[0].transcript if response.results else ""
    except Exception as e:
        text = ""
        print("Google Cloud Speech error:", e)

    # --- Sentiment Analysis ---
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text) if text else {}

    return {
        "pitch": pitch_mean,
        "text": text,
        "sentiment": sentiment
    }
