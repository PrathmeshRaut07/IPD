import numpy as np
import librosa
import os
import pyaudio
import soundfile as sf
from google.cloud import speech
import io
import re
import wave
from dotenv import load_dotenv
FORMAT = pyaudio.paInt16      
CHANNELS = 1                  
RATE = 16000                 
FRAME_DURATION = 30         
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  
VAD_AGGRESSIVENESS = 3        # 0-3 (3 is most aggressive)
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def convert_to_wav(audio_data):
    """
    Converts raw PCM audio data to WAV format.
    """
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(RATE)
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        return wav_buffer.read()
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



import json

def analyze_audio(audio_data):
    """
    Uses Gemini AI to analyze the audio and extract features such as 
    pitch rate, tone, clarity in speaking (0 to 1), and emotion.
    
    Returns a JSON response with the analysis.
    """
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt="""
Analyze this audio clip and return the following metrics in JSON format:  
{  
  "pitch_rate": float (Hz),  
  "tone": string (e.g., "Formal", "Casual", "Persuasive", "Neutral", "Confident"),  
  "clarity_score": float (0-1),  
  "emotion": string (e.g., "Happy", "Sad", "Angry", "Excited", "Calm")  
}  

"""
    print(type(audio_data))
    wav_data = convert_to_wav(audio_data)
    if not isinstance(audio_data, bytes):
        raise ValueError("Audio data must be in bytes format.")
    response_text = model.generate_content([
                prompt,
                {
                    "mime_type": "audio/wav",
                    "data": wav_data
                }
            ])


    # Ensure the response is in JSON format
    try:
        response_text = re.sub(r"```json|```", "", response_text.text).strip()  # Remove backticks
        #print(response_text)  # Debug: Check cleaned JSON string
        analysis_result = json.loads(response_text)  
    except json.JSONDecodeError:
        analysis_result = {"error": "Invalid response format from AI model"}

    return analysis_result

