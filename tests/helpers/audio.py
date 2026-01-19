"""Helper functions for creating test audio files"""

import io
import math
import os
import wave


def create_test_audio_tone(frequency=440, duration=2.0, filename="test_tone.wav"):
    """
    Create a synthetic audio file with a pure sine wave tone.

    Args:
        frequency: Frequency in Hz (default 440Hz = A4 note)
        duration: Duration in seconds
        filename: Output filename

    Returns:
        Path to created WAV file
    """
    sample_rate = 16000  # Higher sample rate for better quality
    num_samples = int(sample_rate * duration)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit audio (standard CD quality)
        wav.setframerate(sample_rate)

        # Generate sine wave samples
        for i in range(num_samples):
            # Generate 16-bit signed integer samples (-32768 to 32767)
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
            # Pack as 16-bit signed integer (little-endian)
            wav.writeframes(value.to_bytes(2, byteorder="little", signed=True))

    # Save to uploads directory (permanent storage, not temp)
    # This ensures files persist after test completes so UI can play them
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, filename)

    with open(filepath, "wb") as f:
        f.write(buffer.getvalue())

    return filepath
