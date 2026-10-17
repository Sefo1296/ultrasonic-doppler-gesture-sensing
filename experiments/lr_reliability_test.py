import time
import os
import numpy as np
import sounddevice as sd
from scipy.signal import welch
from scipy.io.wavfile import write

FS = 48000
FREQ = 19000
DURATION = 2

os.makedirs("captures/lr_tests", exist_ok=True)

def calculate_ratio(recording):
    powers = []

    for ch in [0, 1]:
        signal = recording[:, ch]

        f, p = welch(
            signal,
            fs=FS,
            nperseg=8192
        )

        mask = (
            ((f >= 18800) & (f <= 18975)) |
            ((f >= 19025) & (f <= 19200))
        )

        powers.append(np.sum(p[mask]))

    return 10 * np.log10(
        (powers[0] + 1e-20) /
        (powers[1] + 1e-20)
    )

results = {"LEFT_TO_RIGHT": [], "RIGHT_TO_LEFT": []}

for direction in results:
    print(f"\nTesting {direction}")

    for trial in range(1, 6):
        print(f"Trial {trial}/5 - Prepare your hand")
        time.sleep(3)

        t = np.arange(FS * DURATION) / FS
        tone = 0.02 * np.sin(2 * np.pi * FREQ * t)
        output = np.column_stack((tone, tone)).astype("float32")

        print("SWIPE NOW")

        recording = sd.playrec(
            output,
            samplerate=FS,
            channels=4,
            device=(9, 8),
            blocking=True
        )

        filename = (
            f"captures/lr_tests/{direction}_{trial}.wav"
        )
        write(filename, FS, recording)

        ratio = calculate_ratio(recording)
        results[direction].append(ratio)

        print(f"Ratio: {ratio:.2f} dB")

print("\n===== FINAL RESULTS =====")

for direction, values in results.items():
    print(f"\n{direction}")
    print("Values:", np.round(values, 2))
    print(f"Average: {np.mean(values):.2f} dB")
    print(f"Std deviation: {np.std(values):.2f} dB")
