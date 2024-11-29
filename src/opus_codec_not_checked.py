import os

import ffmpeg

input_file = "./src/common_voice_ar_19065197.mp3"
output_file = "output.opus"
decoded_file = "./src/common_voice_ar_19065197_decoded.wav"

try:
    # Convert WAV to Opus
    print("Encoding WAV to Opus...")
    ffmpeg.input(input_file).output(output_file, acodec="libopus", ab="64k").run(
        overwrite_output=True
    )
    print(f"Encoded Opus file saved to {output_file}")

    # Convert Opus back to WAV
    print("Decoding Opus to WAV...")
    ffmpeg.input(output_file).output(decoded_file).run(overwrite_output=True)
    print(f"Decoded WAV file saved to {decoded_file}")

except ffmpeg.Error as e:
    print(f"An error occurred during processing: {e.stderr.decode()}")
