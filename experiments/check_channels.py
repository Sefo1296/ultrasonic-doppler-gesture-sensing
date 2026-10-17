import numpy as np
import sounddevice as sd

FS = 48000
FREQ = 19000
DURATION = 5

t = np.arange(FS * DURATION) / FS
tone = 0.02 * np.sin(2 * np.pi * FREQ * t)

output = np.column_stack((tone, tone)).astype("float32")

print("Recording all 4 microphone channels...")
print("Move your hand left and right.")

recording = sd.playrec(
    output,
    samplerate=FS,
    channels=4,
    device=(9, 8),
    blocking=True
)

for i in range(4):
    peak = np.max(np.abs(recording[:, i]))
    print(f"Channel {i}: Peak = {peak:.6f}")

print("\nDifferences compared with Channel 0:")

for i in range(1, 4):
    difference = np.max(
        np.abs(recording[:, 0] - recording[:, i])
    )
    print(f"Channel 0 vs {i}: {difference:.9f}")
