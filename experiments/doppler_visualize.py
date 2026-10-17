
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import stft

FS_EXPECTED = 48000
CARRIER = 19000

files = {
    "Baseline (No Movement)": "captures/baseline_still.wav",
    "Hand Movement": "captures/motion_test.wav"
}

os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

spectrograms = []

for title, filename in files.items():

    fs, audio = read(filename)

    if fs != FS_EXPECTED:
        raise ValueError("Unexpected sample rate")

    # Convert recording to floating point
    audio = audio.astype(np.float64) / 32768.0

    # Short-Time Fourier Transform
    f, t, Z = stft(
        audio,
        fs=fs,
        nperseg=8192,
        noverlap=7168,
        boundary=None
    )

    magnitude_db = 20 * np.log10(np.abs(Z) + 1e-12)

    spectrograms.append((title, f, t, magnitude_db))

# Use the same color scale for both recordings
peak = max(np.max(item[3]) for item in spectrograms)

for ax, (title, f, t, magnitude_db) in zip(
    axes, spectrograms
):

    image = ax.pcolormesh(
        t,
        f,
        magnitude_db,
        shading="auto",
        cmap="magma",
        vmin=peak - 80,
        vmax=peak
    )

    ax.axhline(
        CARRIER,
        color="cyan",
        linestyle="--",
        label="19 kHz transmitted tone"
    )

    ax.set_ylim(18500, 19500)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (seconds)")

fig.colorbar(
    image,
    ax=axes,
    label="Magnitude (dB)"
)

fig.suptitle("Ultrasonic Doppler Signal Analysis")

plt.savefig(
    "results/doppler_spectrogram.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Spectrogram saved successfully!")
