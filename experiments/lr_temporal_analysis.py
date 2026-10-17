import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import stft
from scipy.ndimage import median_filter

FREQ = 19000

directions = ["LEFT_TO_RIGHT", "RIGHT_TO_LEFT"]

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for ax, direction in zip(axes, directions):

    for trial in range(1, 6):

        filename = f"captures/lr_tests/{direction}_{trial}.wav"
        fs, audio = read(filename)

        powers = []

        for ch in [0, 1]:
            f, t, Z = stft(
                audio[:, ch],
                fs=fs,
                nperseg=8192,
                noverlap=7168,
                boundary=None
            )

            band = (
                ((f >= FREQ - 200) & (f <= FREQ - 25)) |
                ((f >= FREQ + 25) & (f <= FREQ + 200))
            )

            powers.append(np.sum(np.abs(Z[band])**2, axis=0))

        ratio = 10 * np.log10(
            (powers[0] + 1e-15) /
            (powers[1] + 1e-15)
        )

        ratio = median_filter(ratio, size=5)

        ax.plot(t, ratio, label=f"Trial {trial}")

    ax.set_title(direction.replace("_", " "))
    ax.set_ylabel("Channel 0/1 Ratio (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig("results/lr_temporal_analysis.png", dpi=180)
plt.show()