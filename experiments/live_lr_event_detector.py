import time
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

# Same audio setup used in the recorded experiment
FS = 48000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
INPUT_CHANNELS = 4
LEFT_TONE = 18500
RIGHT_TONE = 19000
AMPLITUDE = 0.02
MIC_CHANNEL = 1  # Second microphone channel (Python index 1)

HISTORY = 2.5
BUFFER_SIZE = int(FS * HISTORY)
FFT_SIZE = 8192
HOP = 2048

# Doppler bands around the 19 kHz RIGHT-speaker signal
UP_BAND = (19035, 19180)
DOWN_BAND = (18820, 18965)

# Conservative starting thresholds; we'll tune them from live tests
MIN_SCORE = 0.20
MIN_FRAMES = 3

buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
lock = Lock()
write_pos = 0
sample_pos = 0
start_time = None

state = {
    "calibration": [],
    "threshold": None,
    "pending": [],
    "event": [],
    "quiet_frames": 0,
    "display": "CALIBRATING",
    "display_until": 0.0,
    "cooldown_until": 0.0,
}


def audio_callback(indata, outdata, frames, time_info, status):
    global write_pos, sample_pos

    n = np.arange(frames) + sample_pos
    outdata[:, 0] = AMPLITUDE * np.sin(2 * np.pi * LEFT_TONE * n / FS)
    outdata[:, 1] = AMPLITUDE * np.sin(2 * np.pi * RIGHT_TONE * n / FS)
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

    if len(event) < MIN_FRAMES:
        return

    # Use the strongest 40% of ACTIVE frames, not one maximum spike.
    count = max(3, int(np.ceil(0.4 * len(event))))
    strongest = sorted(event, key=lambda v: v[0] + v[1], reverse=True)[:count]
    upper = sum(v[0] for v in strongest)
    lower = sum(v[1] for v in strongest)
    score = (upper - lower) / (upper + lower + 1e-30)

    if score > MIN_SCORE:
        label = "RIGHT  (middle -> right speaker)"
    elif score < -MIN_SCORE:
        label = "LEFT  (right speaker -> middle)"
    else:
        label = "UNCERTAIN"

    state["display"] = f"{label}   |   score {score:+.2f}"
    state["display_until"] = now + 1.8
    state["cooldown_until"] = now + 0.6
    print(state["display"], flush=True)


fig, ax = plt.subplots(figsize=(11, 6))
image = ax.imshow(
    np.full((103, 10), -100.0),
    origin="lower",
    aspect="auto",
    extent=[-HISTORY, 0, 18700, 19300],
    cmap="magma",
    vmin=-100,
    vmax=-35,
)
ax.axhline(RIGHT_TONE, color="cyan", linestyle="--", linewidth=1)
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
    visible = (frequencies >= 18700) & (frequencies <= 19300)
    image.set_data(20 * np.log10(magnitude[visible] + 1e-12))

    upper_band = (frequencies >= UP_BAND[0]) & (frequencies <= UP_BAND[1])
    lower_band = (frequencies >= DOWN_BAND[0]) & (frequencies <= DOWN_BAND[1])
    upper = float(np.sum(magnitude[upper_band, -1] ** 2))
    lower = float(np.sum(magnitude[lower_band, -1] ** 2))
    activity = upper + lower

    # Allow output startup to settle; keep your hand still for calibration.
    if elapsed < 4.0:
        if elapsed > 1.0:
            state["calibration"].append(activity)
        state["display"] = f"CALIBRATING: {max(0, 4 - elapsed):.1f}s - keep still"
    else:
        if state["threshold"] is None:
            background = np.asarray(state["calibration"])
            if len(background) < 5:
                state["display"] = "CALIBRATION FAILED - restart"
                ax.set_title(state["display"])
                return (image,)
            state["threshold"] = max(
                12 * np.median(background),
                4 * np.percentile(background, 90),
                1e-20,
            )
            state["display"] = "NO GESTURE"
            print("Calibration done. Swipe between middle and right speaker.")

        if now >= state["cooldown_until"]:
            active = activity > state["threshold"]
            if state["event"]:
                if active:
                    state["event"].append((upper, lower))
                    state["quiet_frames"] = 0
                else:
                    state["quiet_frames"] += 1

                if state["quiet_frames"] >= 3 or len(state["event"]) >= 25:
                    finish_event(now)
            else:
                if active:
                    state["pending"].append((upper, lower))
                    if len(state["pending"]) >= 2:
                        state["event"] = state["pending"][:]
                        state["pending"] = []
                else:
                    state["pending"] = []

        if now > state["display_until"] and not state["event"]:
            state["display"] = "NO GESTURE"
        elif state["event"] and now > state["display_until"]:
            state["display"] = "ANALYZING MOVEMENT..."

    ax.set_title("LIVE MIDDLE / RIGHT DETECTOR | " + state["display"])
    return (image,)


animation = FuncAnimation(
    fig, update, interval=65, blit=False, cache_frame_data=False
)

print("Keep your hand still for the first 4 seconds.")
print("Then swipe MIDDLE -> RIGHT or RIGHT -> MIDDLE.")
print("Only one swipe at a time. Close the window to stop.")

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

print("Live left/right detector stopped.")
