# from Functions.recorder import transcribe_audio
# from Functions.analyzer import analyze_pitch,analyze_audio
# from datetime import datetime
# def process_audio_segment(audio_data):
#     print("\n--- Processing Audio Segment ---")

#     # Get the current timestamp
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # 1. Transcribe the audio segment
#     transcription = transcribe_audio(audio_data)
#     print("Transcription:", transcription)

#     # 2. Analyze pitch from the audio segment
#     json_response = analyze_audio(audio_data)

#     # Extract relevant information
#     pitch_rate = json_response.get('pitch_rate', 'N/A')
#     tone = json_response.get('tone', 'N/A')
#     clarity_score = json_response.get('clarity_score', 'N/A')
#     emotion = json_response.get('emotion', 'N/A')

#     print(f"Pitch Rate: {pitch_rate} Hz")
#     print(f"Tone: {tone}")
#     print(f"Clarity Score: {clarity_score} (0 to 1)")
#     print(f"Emotion: {emotion}")

#     # Store responses in a text file with timestamp
#     with open("audio_analysis.txt", "a") as file:  # Use 'a' to append new entries
#         file.write(f"\n--- Analysis Timestamp: {timestamp} ---\n")
#         file.write(f"Transcription: {transcription}\n")
#         file.write(f"Pitch Rate: {pitch_rate} Hz\n")
#         file.write(f"Tone: {tone}\n")
#         file.write(f"Clarity Score: {clarity_score} (0 to 1)\n")
#         file.write(f"Emotion: {emotion}\n")
#         file.write("-" * 50 + "\n")  # Separator for readability

#     print("Analysis results saved to 'audio_analysis.txt'")


# # Example usage (assuming `transcribe_audio` and `analyze_audio` functions are defined)
# # process_audio_segment(audio_data)
# saver.py
from Functions.recorder import transcribe_audio
from Functions.analyzer import analyze_pitch, analyze_audio
from datetime import datetime

# Import our RL agent and policy functions
from Functions.RL import global_rl_agent, policy_recommendation

# Global variable to store the previous audio metrics
previous_metrics = None

def process_audio_segment(audio_data):
    global previous_metrics
    print("\n--- Processing Audio Segment ---")

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Transcribe the audio segment
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)

    # 2. Analyze the audio segment (using Gemini AI, etc.)
    json_response = analyze_audio(audio_data)

    # Extract relevant information
    pitch_rate = json_response.get('pitch_rate', 'N/A')
    tone = json_response.get('tone', 'N/A')
    clarity_score = json_response.get('clarity_score', 'N/A')
    emotion = json_response.get('emotion', 'N/A')

    print(f"Pitch Rate: {pitch_rate} Hz")
    print(f"Tone: {tone}")
    print(f"Clarity Score: {clarity_score} (0 to 1)")
    print(f"Emotion: {emotion}")

    # Optionally, store responses in a text file with timestamp
    with open("audio_analysis.txt", "a") as file:
        file.write(f"\n--- Analysis Timestamp: {timestamp} ---\n")
        file.write(f"Transcription: {transcription}\n")
        file.write(f"Pitch Rate: {pitch_rate} Hz\n")
        file.write(f"Tone: {tone}\n")
        file.write(f"Clarity Score: {clarity_score} (0 to 1)\n")
        file.write(f"Emotion: {emotion}\n")
        file.write("-" * 50 + "\n")

    # --- RL Agent Integration ---
    # If a previous state exists, use it along with the current metrics
    if previous_metrics is not None:
        action, reward = global_rl_agent.act(previous_metrics, json_response)
        print("RL Action:", action)
        print(f"RL Reward: {reward:.2f}")
        print("RL Recommendation:", policy_recommendation(json_response))
    else:
        print("RL Agent: Waiting for next segment to update policy...")

    # Update the previous_metrics to the current one for the next segment.
    previous_metrics = json_response

    print("Analysis results saved to 'audio_analysis.txt'")
