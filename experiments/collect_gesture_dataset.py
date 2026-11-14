"""Capture labeled ultrasonic Doppler audio for gesture-model training.

Run from the UltrasonicDoppler project root:
    python experiments/collect_gesture_dataset.py

A new dataset session is created under captures/ml_dataset/ (ignored by Git).
Keeps the original working detectors unchanged.
"""

import csv
import json
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ------------ Development-laptop audio configuration ------------
FS = 48_000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
MIC_CHANNELS = 4
LEFT_FREQUENCY = 18_500
RIGHT_FREQUENCY = 19_000
AMPLITUDE = 0.02
BLOCK_SIZE = 1024

# ------------ Capture protocol ------------
REPEATS_PER_CLASS = 5  # First pilot: 25 recordings; increase after inspection
PRE_GO_SECONDS = 0.5   # Baseline samples before the gesture cue
POST_GO_SECONDS = 3.0  # Make one movement during this window
CLASSES = ("LEFT", "RIGHT", "TOWARD", "AWAY", "IDLE")
DATA_ROOT = Path("captures") / "ml_dataset"

# Two alternative paths for each lateral class, to avoid teaching the model
# that one starting position alone determines the movement label.
EXAMPLES = {
    "LEFT": [
        ("Middle of keyboard", "Left speaker", "Palm faces keyboard"),
        ("Right speaker", "Middle of keyboard", "Palm faces keyboard"),
    ],
    "RIGHT": [
        ("Middle of keyboard", "Right speaker", "Palm faces keyboard"),
        ("Left speaker", "Middle of keyboard", "Palm faces keyboard"),
    ],
    "TOWARD": [
        ("Near left speaker", "Closer to left speaker", "Palm faces speaker"),
        ("Near right speaker", "Closer to right speaker", "Palm faces speaker"),
    ],
    "AWAY": [
        ("Close to left speaker", "Farther from left speaker", "Palm faces speaker"),
        ("Close to right speaker", "Farther from right speaker", "Palm faces speaker"),
    ],
    "IDLE": [
        ("Above keyboard middle", "Stay motionless", "Palm faces keyboard"),
        ("Near left speaker", "Stay motionless", "Palm faces speaker"),
        ("Near right speaker", "Stay motionless", "Palm faces speaker"),
    ],
}

FIELDS = [
    "file", "label", "participant", "session", "trial", "hand",
    "start_position", "end_position", "palm_orientation", "speed",
    "mic_channels", "sample_rate_hz", "left_speaker_hz", "right_speaker_hz",
    "recording_seconds", "go_cue_seconds", "peak_abs", "clipped_samples",
    "audio_warnings", "recorded_utc",
]


def safe_id(value, fallback):
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", value.strip())[:30]
    return cleaned or fallback


class AudioCapture:
    """Continuous two-tone output and synchronized four-channel mic capture."""

    def __init__(self):
        self.index = 0
        self.active = None
        self.position = 0
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.warnings = []
        self.stream = None

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.warnings.append(str(status))

        # Phase-continuous stereo tone, active before and during every trial.
        samples = np.arange(frames, dtype=np.float64) + self.index
        outdata[:, 0] = AMPLITUDE * np.sin(
            2.0 * np.pi * LEFT_FREQUENCY * samples / FS
        )
        outdata[:, 1] = AMPLITUDE * np.sin(
            2.0 * np.pi * RIGHT_FREQUENCY * samples / FS
        )
        self.index += frames

        with self.lock:
            if self.active is not None:
                take = min(frames, len(self.active) - self.position)
                if take > 0:
                    self.active[self.position:self.position + take] = indata[:take]
                    self.position += take
                if self.position >= len(self.active):
                    self.active = None
                    self.finished.set()

    def __enter__(self):
        self.stream = sd.Stream(
            samplerate=FS,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            channels=(MIC_CHANNELS, 2),
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self.callback,
        )
        self.stream.start()
        time.sleep(0.4)  # Allow the continuous ultrasonic output to stabilize.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.stop()
        self.stream.close()

    def capture(self):
        count = int(round(FS * (PRE_GO_SECONDS + POST_GO_SECONDS)))
        data = np.empty((count, MIC_CHANNELS), dtype=np.float32)
        self.finished.clear()
        warning_start = len(self.warnings)
        with self.lock:
            self.position = 0
            self.active = data

        # Recording begins first, so every file contains 0.5s before GO.
        time.sleep(PRE_GO_SECONDS)
        print("\n>>> GO! Do ONE gesture, then keep your hand still. <<<", flush=True)
        if not self.finished.wait(timeout=POST_GO_SECONDS + 3.0):
            with self.lock:
                self.active = None
            raise RuntimeError("Audio recording timed out. Check microphone/speakers.")
        print(">>> STOP! Recording saved. <<<", flush=True)
        return data, len(self.warnings) - warning_start


def make_plan(repeats):
    plan = []
    for label in CLASSES:
        variants = EXAMPLES[label]
        for i in range(repeats):
            start, end, orientation = variants[i % len(variants)]
            plan.append((label, start, end, orientation))
    # Shuffle class order so temperature, practice and drift don't match labels.
    random.SystemRandom().shuffle(plan)
    return plan


def main():
    print("ULTRASONIC GESTURE DATASET CAPTURE")
    print("5 labels: LEFT, RIGHT, TOWARD, AWAY, IDLE (no movement)")
    print("Left speaker = 18.5 kHz; right speaker = 19.0 kHz")
    print("Microphone: all 4 channels; 48 kHz; saved as WAV PCM16")
    print("Working detector scripts will NOT be changed.\n")

    participant = safe_id(input("Anonymous participant ID [P01]: ") or "P01", "P01")
    hand = (input("Hand used throughout this session [RIGHT]: ") or "RIGHT").upper().strip()
    if hand not in ("RIGHT", "LEFT"):
        raise ValueError("Hand must be RIGHT or LEFT")
    speed = (input("Swipe speed [NORMAL]: ") or "NORMAL").upper().strip()
    if speed not in ("SLOW", "NORMAL", "FAST"):
        raise ValueError("Speed must be SLOW, NORMAL, or FAST")

    session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + participant
    folder = DATA_ROOT / session
    folder.mkdir(parents=True, exist_ok=False)
    plan = make_plan(REPEATS_PER_CLASS)

    metadata = {
        "session": session,
        "participant": participant,
        "hand": hand,
        "speed": speed,
        "labels": list(CLASSES),
        "trial_count": len(plan),
        "fs_hz": FS,
        "mic_channels": MIC_CHANNELS,
        "input_device_index": INPUT_DEVICE,
        "output_device_index": OUTPUT_DEVICE,
        "left_speaker_hz": LEFT_FREQUENCY,
        "right_speaker_hz": RIGHT_FREQUENCY,
        "output_amplitude": AMPLITUDE,
        "pre_go_seconds": PRE_GO_SECONDS,
        "post_go_seconds": POST_GO_SECONDS,
        "format": "interleaved 4-channel signed 16-bit PCM WAV",
        "notes": "Files include a quiet pre-cue interval. A single gesture is requested after GO. Labels describe the intended gesture, not an independently verified ground truth.",
    }
    (folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaving into: {folder.resolve()}")
    print(f"Pilot: {len(plan)} recordings ({REPEATS_PER_CLASS} per label)")
    print("Every trial announces the label and START/END position BEFORE recording.")
    print("Do not run other Doppler/audio programs at the same time.")

    done = 0
    csv_path = folder / "labels.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f, AudioCapture() as audio:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            f.flush()
            for trial, (label, start, end, orientation) in enumerate(plan, 1):
                print("\n" + "=" * 58)
                print(f"TRIAL {trial}/{len(plan)} | LABEL: {label}")
                print(f"START: {start}\nEND:   {end}\nHAND:  {hand}")
                print(f"PALM:  {orientation}\nSPEED: {speed}")
                if label == "IDLE":
                    print("IMPORTANT: Keep your hand in place for the ENTIRE recording.")
                else:
                    print("IMPORTANT: Make ONE movement at GO and then stop.")
                print("Recording has NOT started yet.")
                command = input("Press ENTER when ready (or Q + ENTER to quit): ").strip().lower()
                if command == "q":
                    break
                for n in (3, 2, 1):
                    print(f"Starting in {n}...", flush=True)
                    time.sleep(1)

                raw, warning_count = audio.capture()
                peak_abs = float(np.max(np.abs(raw)))
                clipped_samples = int(np.count_nonzero(np.abs(raw) >= 0.999))
                filename = f"{trial:03d}_{label}.wav"
                pcm = (np.clip(raw, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(folder / filename, FS, pcm)
                writer.writerow({
                    "file": filename,
                    "label": label,
                    "participant": participant,
                    "session": session,
                    "trial": trial,
                    "hand": hand,
                    "start_position": start,
                    "end_position": end,
                    "palm_orientation": orientation,
                    "speed": speed,
                    "mic_channels": MIC_CHANNELS,
                    "sample_rate_hz": FS,
                    "left_speaker_hz": LEFT_FREQUENCY,
                    "right_speaker_hz": RIGHT_FREQUENCY,
                    "recording_seconds": PRE_GO_SECONDS + POST_GO_SECONDS,
                    "go_cue_seconds": PRE_GO_SECONDS,
                    "peak_abs": f"{peak_abs:.6f}",
                    "clipped_samples": clipped_samples,
                    "audio_warnings": warning_count,
                    "recorded_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
                f.flush()
                done += 1
                print(f"Saved {filename} | peak={peak_abs:.4f} | audio warnings={warning_count}")
                if clipped_samples:
                    print("WARNING: Audio clipped; consider excluding this recording.")
                if warning_count:
                    print("WARNING: Audio status warning; inspect this recording.")
    finally:
        print(f"\nCOMPLETE: {done} WAV files. Metadata: {csv_path}")
        print("These are local in captures/ml_dataset/ (ignored by Git).")
        print("Share the session folder with your model-training teammate.")


if __name__ == "__main__":
    main()
