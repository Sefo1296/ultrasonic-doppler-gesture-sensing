"""Capture labeled ultrasonic Doppler audio for gesture-model training (v2).

Run from the UltrasonicDoppler project root:
    python experiments/collect_gesture_dataset_v2.py

A new dataset session is created under captures/ml_dataset/ (ignored by Git).
Keeps the original working detectors unchanged.
"""

import csv
import json
import random
import re
import threading
import time
from datetime import datetime, timezone
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
REPEATS_PER_CLASS = 5  # Verification session: 25 recordings; increase later
PRE_GO_SECONDS = 0.5   # Baseline samples before the gesture cue
POST_GO_SECONDS = 3.0  # Make one movement during this window
CLASSES = ("LEFT", "RIGHT", "TOWARD", "AWAY", "IDLE")
DATA_ROOT = Path("captures") / "ml_dataset"
PALM_ORIENTATION = "Vertical; palm faces RIGHT"

# Two alternative paths for each lateral class, to avoid teaching the model
# that one starting position alone determines the movement label.
EXAMPLES = {
    "LEFT": [
        ("Middle of keyboard", "Left speaker", PALM_ORIENTATION),
        ("Right speaker", "Middle of keyboard", PALM_ORIENTATION),
    ],
    "RIGHT": [
        ("Middle of keyboard", "Right speaker", PALM_ORIENTATION),
        ("Left speaker", "Middle of keyboard", PALM_ORIENTATION),
    ],
    "TOWARD": [
        ("Near left speaker", "Closer to left speaker", PALM_ORIENTATION),
        ("Near right speaker", "Closer to right speaker", PALM_ORIENTATION),
    ],
    "AWAY": [
        ("Close to left speaker", "Farther from left speaker", PALM_ORIENTATION),
        ("Close to right speaker", "Farther from right speaker", PALM_ORIENTATION),
    ],
    "IDLE": [
        ("Above keyboard middle", "Stay motionless", PALM_ORIENTATION),
        ("Near left speaker", "Stay motionless", PALM_ORIENTATION),
        ("Near right speaker", "Stay motionless", PALM_ORIENTATION),
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


def ask_hand():
    """Require the actual hand used for this recording: never assume a default."""
    while True:
        answer = input("HAND FOR THIS TRIAL (L=LEFT / R=RIGHT; required): ").strip().upper()
        if answer in ("L", "LEFT"):
            return "LEFT"
        if answer in ("R", "RIGHT"):
            return "RIGHT"
        print("Please type L or R. We won't guess the hand used.")


def ask_keep_recording():
    """Let the operator discard a misperformed recording before saving it."""
    while True:
        answer = input("Was that trial correct? [ENTER=save / R=redo / Q=quit]: ").strip().upper()
        if answer == "":
            return "save"
        if answer == "R":
            return "redo"
        if answer == "Q":
            return "quit"
        print("Press ENTER to save, R to redo, or Q to quit.")


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
    speed = (input("Target swipe speed [NORMAL]: ") or "NORMAL").upper().strip()
    if speed not in ("SLOW", "NORMAL", "FAST"):
        raise ValueError("Speed must be SLOW, NORMAL, or FAST")

    session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + participant
    folder = DATA_ROOT / session
    folder.mkdir(parents=True, exist_ok=False)
    plan = make_plan(REPEATS_PER_CLASS)

    metadata = {
        "session": session,
        "participant": participant,
        "hand": "RECORDED_PER_TRIAL_IN_LABELS_CSV",
        "palm_orientation": PALM_ORIENTATION,
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
        "notes": "Files include a 0.5s pre-GO baseline. A single gesture is requested after GO. Hand is explicitly recorded for EACH trial. The palm orientation is participant-reported. Target speed is an instruction, not a measured speed. Gesture labels describe intended movement and have not been independently verified.",
    }
    (folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaving into: {folder.resolve()}")
    print(f"Pilot: {len(plan)} recordings ({REPEATS_PER_CLASS} per label)")
    print("Every trial announces the label and START/END position BEFORE recording.")
    print("You MUST type the hand you will use (L/R) for EACH trial.")
    print(f"Palm orientation for EVERY trial: {PALM_ORIENTATION}.")
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
                print(f"START: {start}\nEND:   {end}")
                print(f"PALM:  {orientation}\nTARGET SPEED: {speed}")
                hand = ask_hand()
                print(f"HAND CONFIRMED: {hand}")
                if label == "IDLE":
                    print("IMPORTANT: Keep your hand in place for the ENTIRE recording.")
                else:
                    print("IMPORTANT: Make ONE movement at GO and then stop.")
                print("Recording has NOT started yet.")
                command = input("Press ENTER when ready (or Q + ENTER to quit): ").strip().lower()
                if command == "q":
                    break
                while True:
                    for n in (3, 2, 1):
                        print(f"Starting in {n}...", flush=True)
                        time.sleep(1)

                    raw, warning_count = audio.capture()
                    peak_abs = float(np.max(np.abs(raw)))
                    clipped_samples = int(np.count_nonzero(np.abs(raw) >= 0.999))
                    print(f"Preview: peak={peak_abs:.4f}, clipping={clipped_samples}, audio warnings={warning_count}")
                    choice = ask_keep_recording()
                    if choice == "redo":
                        print("Discarded unsaved take. Reset to the START position and try again.")
                        continue
                    if choice == "quit":
                        print("Stopped without saving the last take.")
                        break
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
                    "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    })
                    f.flush()
                    done += 1
                    print(f"Saved {filename} | hand={hand} | peak={peak_abs:.4f} | audio warnings={warning_count}")
                    if clipped_samples:
                        print("WARNING: Audio clipped; consider excluding this recording.")
                    if warning_count:
                        print("WARNING: Audio status warning; inspect this recording.")
                    break
                if choice == "quit":
                    break
    finally:
        print(f"\nCOMPLETE: {done} WAV files. Metadata: {csv_path}")
        print("These are local in captures/ml_dataset/ (ignored by Git).")
        print("Share the session folder with your model-training teammate.")


if __name__ == "__main__":
    main()
