import ffmpeg

input_file = "./src/common_voice_ar_19065197.mp3"
decoded_file = "./src/common_voice_ar_19065197_decoded.wav"

try:
    # Convert input audio to Opus and capture output in memory
    print("Encoding audio to Opus in memory...")
    encoded_opus, stderr = (
        ffmpeg.input(input_file)
        .output("pipe:", format="opus", acodec="libopus", ab="64k")
        .run(capture_stdout=True, capture_stderr=True)
    )
    print("Encoding completed.")

    # Decode Opus from memory and save as WAV
    print("Decoding Opus to WAV...")
    decoded_pcm, stderr = (
        ffmpeg.input("pipe:0", format="ogg")  # Opus is often encapsulated in Ogg format
        .output(decoded_file, format="wav")
        .run(input=encoded_opus, capture_stdout=True, capture_stderr=True)
    )
    print(f"Decoded WAV file saved to {decoded_file}")

except ffmpeg.Error as e:
    stderr_output = e.stderr.decode() if e.stderr else "No stderr captured"
    print(f"An error occurred during processing: {stderr_output}")
