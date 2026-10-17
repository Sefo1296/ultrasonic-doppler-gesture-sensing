import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io.wavfile import write

FS = 48000
FREQ = 19000
DURATION = 8

os.makedirs("results", exist_ok=True)
os.makedirs("captures", exist_ok=True)

# Generate 19 kHz tone
t = np.arange(FS * DURATION) / FS
tone = 0.02 * np.sin(2 * np.pi * FREQ * t)

audio_out = np.column_stack((tone, tone)).astype("float32")

print("Recording for 8 seconds...")
print("0-2s: Keep hand still")
print("2-4s: Swipe LEFT to RIGHT")
print("4-5s: Keep hand still")
print("5-7s: Swipe RIGHT to LEFT")
print("7-8s: Keep hand still")

recording = sd.playrec(
    audio_out,
    samplerate=FS,
    channels=4,
    device=(9, 8),
    blocking=True
)

# Save all 4 channels without clipping to int16
write("captures/four_channels.wav", FS, recording)

fig, axes = plt.subplots(
    4, 1, figsize=(12, 10),
    sharex=True, sharey=True
)

for ch in range(4):
    frequencies, times, spectrum = spectrogram(
        recording[:, ch],
        fs=FS,
        nperseg=8192,
        noverlap=7168,
        mode="magnitude"
    )

    db = 20 * np.log10(spectrum + 1e-12)

    mask = (
        (frequencies >= 18700) &
        (frequencies <= 19300)
    )

    axes[ch].pcolormesh(
        times,
        frequencies[mask],
        db[mask],
        shading="auto",
        cmap="magma",
        vmin=-100,
        vmax=-35
    )

    axes[ch].axhline(
        FREQ, color="cyan",
        linestyle="--"
    )

    axes[ch].set_title(f"Microphone Channel {ch}")
    axes[ch].set_ylabel("Hz")

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig(
    "results/four_channel_comparison.png",
    dpi=180
)
plt.show()

print("Comparison saved successfully.")
