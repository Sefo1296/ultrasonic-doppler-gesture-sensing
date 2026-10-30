"""Experimental left-side ultrasonic gesture detector.

Test zone: middle of keyboard <-> left speaker.
This file is intentionally independent of the working right-side detector.
"""

import time
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

# Laptop configuration (same devices and tones as the previous left-side test)
FS = 48000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
INPUT_CHANNELS = 4
MIC_CHANNEL = 1  # Second microphone input
LEFT_TONE = 18500
RIGHT_TONE = 19000
AMPLITUDE = 0.02

HISTORY = 2.5
BUFFER_SIZE = int(FS * HISTORY)
FFT_SIZE = 8192
HOP = 2048

# Analyze the Doppler signal from the LEFT speaker only.
UP_BAND = (18535, 18680)
DOWN_BAND = (18320, 18465)
DISPLAY_BAND = (18200, 18800)

CALIBRATION_SECONDS = 4.0
MIN_SCORE = 0.25  # A lower absolute score is too ambiguous to classify.
MIN_EVENT_FRAMES = 3
START_FRAMES = 3  # Consecutive activity checks needed to start an event
STOP_FRAMES = 3
COOLDOWN_SECONDS = 0.65

# Experimentally reduced from the previous 12x median / 4x P90 threshold.
# We require a multi-frame event to help reject noise bursts.
BACKGROUND_MEDIAN_MULTIPLIER = 6.0
BACKGROUND_P90_MULTIPLIER = 2.5

buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
lock = Lock()
write_pos = 0
sample_pos = 0
start_time = None

state = {
    "calibration": [],
    "threshold": None,
    "noise_median": None,
    "pending": [],
    "event": [],
    "quiet_frames": 0,
    "display": "CALIBRATING",
    "display_until": 0.0,
    "cooldown_until": 0.0,
    "accepted_count": 0,
    "uncertain_count": 0,
}


def audio_callback(indata, outdata, frames, time_info, status):
    global write_pos, sample_pos
    samples = np.arange(frames) + sample_pos

    # Maintain the same dual-tone audio setup to change only detection logic.
    outdata[:, 0] = AMPLITUDE * np.sin(2 * np.pi * LEFT_TONE * samples / FS)
    outdata[:, 1] = AMPLITUDE * np.sin(2 * np.pi * RIGHT_TONE * samples / FS)
    sample_pos += frames

    incoming = indata[:, MIC_CHANNEL]
    with lock:
        first = min(frames, BUFFER_SIZE - write_pos)
        buffer[write_pos:write_pos + first] = incoming[:first]
        if frames > first:
            buffer[:frames - first] = incoming[first:]
        write_pos = (write_pos + frames) % BUFFER_SIZE


def finish_event(now):
    event = state["event"]
    state["event"] = []
    state["pending"] = []
    state["quiet_frames"] = 0

    if len(event) < MIN_EVENT_FRAMES:
        return

    # Prefer a short, strong movement segment, not the loudest single frame.
    count = max(3, int(np.ceil(0.40 * len(event))))
    strongest = sorted(event, key=lambda v: v[0] + v[1], reverse=True)[:count]
    upper = sum(v[0] for v in strongest)
    lower = sum(v[1] for v in strongest)
    score = (upper - lower) / (upper + lower + 1e-30)

    peak_activity = max(u + d for u, d in event)
    signal_ratio = peak_activity / max(state["threshold"], 1e-30)

    if score >= MIN_SCORE:
        label = "LEFT  (middle -> left speaker)"
        state["accepted_count"] += 1
    elif score <= -MIN_SCORE:
        label = "RIGHT  (left speaker -> middle)"
        state["accepted_count"] += 1
    else:
        label = "UNCERTAIN"
        state["uncertain_count"] += 1

    state["display"] = f"{label} | score {score:+.2f} | peak {signal_ratio:.1f}x gate"
    state["display_until"] = now + 1.8
    state["cooldown_until"] = now + COOLDOWN_SECONDS
    print(state["display"], flush=True)


fig, ax = plt.subplots(figsize=(11, 6))
image = ax.imshow(
    np.full((103, 10), -100.0),
    origin="lower",
    aspect="auto",
    extent=[-HISTORY, 0, *DISPLAY_BAND],
    cmap="magma",
    vmin=-100,
    vmax=-35,
)
ax.axhline(LEFT_TONE, color="cyan", linestyle="--", linewidth=1)
ax.set_xlabel("Time relative to now (seconds)")
ax.set_ylabel("Frequency (Hz)")
plt.colorbar(image, ax=ax, label="Magnitude (dB)")


def update(_):
    now = time.monotonic()
    elapsed = now - start_time

    with lock:
        current = np.concatenate((buffer[write_pos:], buffer[:write_pos]))

    frequencies, _, magnitude = spectrogram(
        current,
        fs=FS,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP,
        window="hann",
        mode="magnitude",
    )
    visible = (frequencies >= DISPLAY_BAND[0]) & (frequencies <= DISPLAY_BAND[1])
    image.set_data(20 * np.log10(magnitude[visible] + 1e-12))

    upper_band = (frequencies >= UP_BAND[0]) & (frequencies <= UP_BAND[1])
    lower_band = (frequencies >= DOWN_BAND[0]) & (frequencies <= DOWN_BAND[1])
    upper = float(np.sum(magnitude[upper_band, -1] ** 2))
    lower = float(np.sum(magnitude[lower_band, -1] ** 2))
    activity = upper + lower

    if elapsed < CALIBRATION_SECONDS:
        if elapsed > 1.0:
            state["calibration"].append(activity)
        state["display"] = (
            f"CALIBRATING: {max(0, CALIBRATION_SECONDS - elapsed):.1f}s | keep still"
        )
    else:
        if state["threshold"] is None:
            background = np.asarray(state["calibration"])
            if len(background) < 5:
                ax.set_title("CALIBRATION FAILED - restart")
                return (image,)
            median = float(np.median(background))
            p90 = float(np.percentile(background, 90))
            state["noise_median"] = median
            state["threshold"] = max(
                BACKGROUND_MEDIAN_MULTIPLIER * median,
                BACKGROUND_P90_MULTIPLIER * p90,
                1e-20,
            )
            state["display"] = "NO GESTURE"
            print("Calibration done. Use MIDDLE <-> LEFT speaker only.")
            print("This version displays motion energy as a multiple of the detection gate.")
            print("Pause at least 1 second between swipes.\n")

        threshold = state["threshold"]
        if now >= state["cooldown_until"]:
            active = activity > threshold
            if state["event"]:
                if active:
                    state["event"].append((upper, lower))
                    state["quiet_frames"] = 0
                else:
                    state["quiet_frames"] += 1

                if state["quiet_frames"] >= STOP_FRAMES or len(state["event"]) >= 25:
                    finish_event(now)
            elif active:
                state["pending"].append((upper, lower))
                if len(state["pending"]) >= START_FRAMES:
                    state["event"] = state["pending"][:]
                    state["pending"] = []
            else:
                state["pending"] = []

        if now > state["display_until"]:
            state["display"] = (
                "ANALYZING MOVEMENT..." if state["event"] else "NO GESTURE"
            )

        # Unlike the Doppler direction score, this is a real signal-to-gate reading.
        gate_ratio = activity / threshold
        ax.set_title(
            f"LIVE MIDDLE / LEFT | {state['display']}\n"
            f"Current movement energy: {gate_ratio:.2f}x detection gate"
        )
        return (image,)

    ax.set_title("LIVE MIDDLE / LEFT | " + state["display"])
    return (image,)


animation = FuncAnimation(
    fig, update, interval=65, blit=False, cache_frame_data=False
)

print("LEFT-SIDE EXPERIMENT (improved threshold + diagnostics)")
print("Keep your hand still for the first four seconds.")
print("Then swipe MIDDLE -> LEFT or LEFT -> MIDDLE.")
print("Close other Doppler applications before starting.")

with sd.Stream(
    samplerate=FS,
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    channels=(INPUT_CHANNELS, 2),
    dtype="float32",
    blocksize=1024,
    callback=audio_callback,
):
    start_time = time.monotonic()
    plt.show()

print(f"Finished. Classified: {state['accepted_count']}; uncertain: {state['uncertain_count']}")
