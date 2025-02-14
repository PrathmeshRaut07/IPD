from Functions.recorder import transcribe_audio
from Functions.analyzer import analyze_pitch,analyze_sentiment
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