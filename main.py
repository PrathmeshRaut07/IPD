import os
import pyaudio
from Functions.recorder import record_speech
from Functions.saver import save_audio
from Functions.analyzer import analyze_audio
key_path='stoked-forest-447811-u4-e25fdb4e6e1c.json'

# -------------------------------
# Configuration Parameters
# -------------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Silence detection parameters (adjust these based on your environment)
SILENCE_THRESHOLD = 1000  # RMS threshold for silence
SILENCE_CHUNKS = 20       # Number of consecutive silent chunks to determine end-of-speech

# Folder to store audio files
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def main():
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    segment_count = 1

    try:
        while True:
            # Record audio until silence is detected
            frames = record_speech(stream, CHUNK, SILENCE_THRESHOLD, SILENCE_CHUNKS)
            
            if frames:
                filename = f"audio_segment_{segment_count}.wav"
                filepath = os.path.join(AUDIO_DIR, filename)
                
                # Save the recorded audio
                save_audio(frames, filepath, audio_interface, CHANNELS, FORMAT, RATE)
                print(f"Saved audio segment as: {filepath}")
                
                # Analyze the saved audio
                results = analyze_audio(filepath, RATE)
                print("\n--- Analysis Results ---")
                print(f"Pitch (Hz): {results['pitch']}")
                print(f"Text Spoken: {results['text']}")
                print(f"Sentiment: {results['sentiment']}")
                print("------------------------\n")
                
                segment_count += 1
                print("Ready for the next segment. Speak when you are ready...\n")
            else:
                print("No audio recorded.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

if __name__ == '__main__':
    main()
