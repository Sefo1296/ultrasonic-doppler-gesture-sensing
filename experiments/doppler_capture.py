
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Audio settings
FS = 48000
FREQ = 19000
DURATION = 5

INPUT_DEVICE = 9
OUTPUT_DEVICE = 8

# Create folder for recordings
os.makedirs("captures", exist_ok=True)

# Generate 19 kHz ultrasonic tone
t = np.arange(FS * DURATION) / FS
tone = 0.02 * np.sin(2 * np.pi * FREQ * t)

# Send tone through both speakers
audio_out = np.column_stack((tone, tone)).astype("float32")

print("Recording hand movement for 5 seconds...")
print("Move your hand toward and away from the laptop.")
print("Keep speaker volume at approximately 25%.")

# Play tone and record microphone simultaneously
recording = sd.playrec(
    audio_out,
    samplerate=FS,
    channels=4,
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    blocking=True
)

# Extract microphone channel 0
channel0 = recording[:, 0]

# Convert to 16-bit WAV format
wav_data = np.int16(np.clip(channel0, -1, 1) * 32767)

# Save recording
filename = "captures/motion_test.wav"
write(filename, FS, wav_data)

print("\nRecording completed.")
print("Saved:", filename)
print("Peak level:", np.max(np.abs(channel0)))
