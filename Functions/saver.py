import wave

def save_audio(frames, filepath, audio_interface, channels, format, rate):
    """
    Saves the recorded audio frames as a WAV file at the specified filepath.
    """
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio_interface.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
