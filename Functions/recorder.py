import collections
import pyaudio
import webrtcvad
import numpy as np
import os
import librosa
from google.cloud import speech
from textblob import TextBlob
key_path=r"stoked-forest-447811-u4-e25fdb4e6e1c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

FORMAT = pyaudio.paInt16      
CHANNELS = 1                  
RATE = 16000                 
FRAME_DURATION = 30         
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  
VAD_AGGRESSIVENESS = 3        # 0-3 (3 is most aggressive)

p = pyaudio.PyAudio()
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
NUM_PADDING_FRAMES = int(300 / FRAME_DURATION)

def record_audio_stream():
    """
    Continuously records from the microphone and yields an audio segment (as bytes)
    once speech has ended.
    """
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )
    print("Listening... (press Ctrl+C to stop)")
    
    ring_buffer = collections.deque(maxlen=NUM_PADDING_FRAMES)
    triggered = False
    voiced_frames = []

    try:
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            
            if not triggered:
                # Buffer the frames until speech is detected
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech_flag in ring_buffer if speech_flag])
                # If most frames in the buffer contain speech, trigger recording
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    # Add all buffered frames to voiced_frames
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # Append current frame and add it to the ring buffer as well
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                # If most frames are silence, consider the segment ended
                num_unvoiced = len([f for f, speech_flag in ring_buffer if not speech_flag])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    yield b"".join(voiced_frames)
                    triggered = False
                    ring_buffer.clear()
                    voiced_frames = []
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()