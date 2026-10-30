
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram
from pathlib import Path
from datetime import datetime
import time

# Audio settings
FS = 48000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
INPUT_CHANNELS = 4

LEFT_FREQ = 18500
RIGHT_FREQ = 19000
AMPLITUDE = 0.02

DURATION = 3
REPEATS = 5

# Left and right speakers transmit different frequencies
t = np.arange(int(FS * DURATION)) / FS

output = np.column_stack([
    AMPLITUDE * np.sin(2 * np.pi * LEFT_FREQ * t),
    AMPLITUDE * np.sin(2 * np.pi * RIGHT_FREQ * t)
]).astype(np.float32)


def record_trial(title, instructions):
    print("\n" + "=" * 55)
    print(f"NEXT TRIAL: {title}")
    print("=" * 55)
    print(instructions)
    print("\nRecording has NOT started.")

    input("\nPress ENTER when you are ready...")

    for seconds in (3, 2, 1):
        print(f"Starting in {seconds}...", flush=True)
        time.sleep(1)

    print("\n>>> GO! MOVE YOUR HAND NOW <<<", flush=True)

    recording = sd.playrec(
        output,
        samplerate=FS,
        channels=INPUT_CHANNELS,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        blocking=True
    )

    print(">>> STOP! Recording complete. <<<")

    return recording.copy()


def measure_bands(recording):
    frequencies, times, power = spectrogram(
        recording[:, 0],
        fs=FS,
        nperseg=8192,
        noverlap=6144,
        mode="psd"
    )

    valid_time = (times >= 0.25) & (times <= 2.75)
    results = {}

    for name, carrier in [
        ("LEFT", LEFT_FREQ),
        ("RIGHT", RIGHT_FREQ)
    ]:
        upper = (
            (frequencies >= carrier + 30) &
            (frequencies <= carrier + 170)
        )

        lower = (
            (frequencies >= carrier - 170) &
            (frequencies <= carrier - 30)
        )

        results[name] = (
            power[upper][:, valid_time].sum(axis=0),
            power[lower][:, valid_time].sum(axis=0)
        )

    return results


def calculate_score(trial, baseline):
    scores = {}

    for speaker in ("LEFT", "RIGHT"):
        up, down = trial[speaker]
        idle_up, idle_down = baseline[speaker]

        base_up = np.median(idle_up)
        base_down = np.median(idle_down)

        idle_variation = (
            np.abs(idle_up - base_up) +
            np.abs(idle_down - base_down)
        )

        threshold = 3 * np.percentile(idle_variation, 99)

        extra_up = np.maximum(up - base_up, 0)
        extra_down = np.maximum(down - base_down, 0)

        activity = extra_up + extra_down
        active = activity > max(threshold, 1e-20)

        if np.count_nonzero(active) < 6:
            scores[speaker] = None
            continue

        direction_score = (
            (extra_up - extra_down) /
            (activity + 1e-30)
        )

        scores[speaker] = float(
            np.median(direction_score[active])
        )

    return scores


def display_score(value):
    if value is None:
        return "NO SIGNAL"
    return f"{value:+.3f}"


print("\nDUAL-SPEAKER LEFT/RIGHT EXPERIMENT")
print("Left speaker:  18.5 kHz")
print("Right speaker: 19.0 kHz")
print("5 trials per direction")
print("3 seconds per recording")
print("\nKeep your palm facing the keyboard.")
print("Try to keep your swipe speed consistent.")

recordings = {}

# Baseline
rest = record_trial(
    "BASELINE CALIBRATION",
    "Keep your hand completely still and away "
    "from the speakers for 3 seconds."
)

recordings["baseline"] = rest
baseline = measure_bands(rest)

results = {
    "L_TO_R": [],
    "R_TO_L": []
}

# Ten gesture trials
for number in range(1, REPEATS + 1):

    for label, direction, instructions in [
        (
            "L_TO_R",
            "LEFT TO RIGHT",
            "Start above the LEFT speaker.\n"
            "When GO appears, swipe across to the RIGHT speaker."
        ),
        (
            "R_TO_L",
            "RIGHT TO LEFT",
            "Start above the RIGHT speaker.\n"
            "When GO appears, swipe across to the LEFT speaker."
        )
    ]:

        recording = record_trial(
            f"{number}/{REPEATS} - {direction}",
            instructions
        )

        recordings[f"{label}_{number}"] = recording

        trial = measure_bands(recording)
        scores = calculate_score(trial, baseline)

        results[label].append(scores)

        print("\nRESULT:")
        print(f"Left speaker score:  {display_score(scores['LEFT'])}")
        print(f"Right speaker score: {display_score(scores['RIGHT'])}")


print("\n" + "=" * 55)
print("CONSISTENCY SUMMARY")
print("=" * 55)

for speaker in ("LEFT", "RIGHT"):
    print(f"\n{speaker} SPEAKER SIGNAL")

    for label in ("L_TO_R", "R_TO_L"):

        values = [
            row[speaker]
            for row in results[label]
            if row[speaker] is not None
        ]

        if not values:
            print(f"{label}: No reliable signal")
            continue

        print(
            f"{label}: "
            f"median={np.median(values):+.3f}, "
            f"min={min(values):+.3f}, "
            f"max={max(values):+.3f}, "
            f"valid={len(values)}/{REPEATS}"
        )


# Save recordings for later analysis
Path("captures").mkdir(exist_ok=True)

filename = (
    "captures/lr_dual_tone_" +
    datetime.now().strftime("%Y%m%d_%H%M%S") +
    ".npz"
)

np.savez_compressed(filename, **recordings)

print(f"\nSaved recordings: {filename}")
print("Experiment complete.")
