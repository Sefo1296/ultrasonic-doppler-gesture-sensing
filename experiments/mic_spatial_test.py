
import time
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, correlate
from scipy.io.wavfile import write
import os

FS = 48000
MIC = 9

os.makedirs("captures", exist_ok=True)

sos = butter(
    3, [500, 6000],
    btype="bandpass",
    fs=FS,
    output="sos"
)

def test_side(side):
    print(f"\nPrepare for {side} test")

    for i in [3, 2, 1]:
        print(i)
        time.sleep(1)

    print("RECORDING - Snap your fingers once!")

    recording = sd.rec(
        3 * FS,
        samplerate=FS,
        channels=4,
        device=MIC,
        dtype="float32"
    )
    sd.wait()

    write(f"captures/mic_{side}.wav", FS, recording)

    # Analyze the first two channels
    filtered = sosfiltfilt(
        sos, recording[:, :2], axis=0
    )

    # Find the snap transient
    start = int(0.4 * FS)
    end = int(2.6 * FS)

    level = np.max(np.abs(filtered), axis=1)
    peak = start + np.argmax(level[start:end])

    # Analyze a short window around the snap
    a = filtered[peak-1440:peak+2400, 0]
    b = filtered[peak-1440:peak+2400, 1]

    if len(a) != 3840 or len(b) != 3840:
        print("Snap too close to recording boundary.")
        return

    a = a - np.mean(a)
    b = b - np.mean(b)

    correlation = correlate(a, b, mode="full", method="fft")
    lags = np.arange(-len(b)+1, len(a))

    mask = np.abs(lags) <= 60
    index = np.argmax(np.abs(correlation[mask]))

    delay_samples = lags[mask][index]
    delay_us = delay_samples / FS * 1e6

    similarity = (
        np.abs(correlation[mask][index]) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )

    print(f"Delay (Ch0 relative to Ch1): {delay_us:+.1f} us")
    print(f"Signal similarity: {similarity:.3f}")

test_side("LEFT")
test_side("RIGHT")
