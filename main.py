import collections
import pyaudio
import webrtcvad
import numpy as np
import os
import librosa
from google.cloud import speech
from textblob import TextBlob
from Functions.recorder import record_audio_stream
from Functions.saver import process_audio_segment
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


# -----------------------------
# Processing Pipeline
# -----------------------------


# -----------------------------
# Main Loop
# -----------------------------
def main():
    for segment in record_audio_stream():
        process_audio_segment(segment)

if __name__ == "__main__":
    main()
