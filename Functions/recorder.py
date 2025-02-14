import audioop

def record_speech(stream, chunk, silence_threshold, silence_chunks):
    """
    Records audio from a given stream until a period of silence is detected.
    Returns a list of audio frames.
    """
    print("Listening for speech... (please speak)")
    frames = []
    silence_counter = 0

    while True:
        data = stream.read(chunk)
        frames.append(data)

        # Compute RMS (volume level)
        rms = audioop.rms(data, 2)  # 2 bytes per sample for paInt16
        if rms < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0

        # End recording if we've had enough silent chunks (and some speech was captured)
        if silence_counter > silence_chunks and len(frames) > silence_chunks:
            break

    return frames
