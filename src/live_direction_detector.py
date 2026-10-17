import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from threading import Lock

# -----------------------------
# Audio configuration
# -----------------------------
FS = 48000
FREQ = 19000
AMPLITUDE = 0.02

INPUT_DEVICE = 9
OUTPUT_DEVICE = 8

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 2

# -----------------------------
# Spectrogram / display settings
# -----------------------------
HISTORY = 4                      # seconds displayed
BUFFER_SIZE = FS * HISTORY
FFT_SIZE = 8192
HOP = 2048

LOW_FREQ = 18700
HIGH_FREQ = 19300

# Analysis bands
CENTER_GUARD = 20               # ignore +-20 Hz around carrier
SIDE_BAND = 180                 # examine up to 180 Hz above/below

# Decision thresholds
MOTION_THRESHOLD_DB = -78       # minimum sideband level to call it motion
DIRECTION_MARGIN_DB = 2.5       # how much stronger one side must be

# -----------------------------
# Shared buffer
# -----------------------------
buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
lock = Lock()
write_position = 0
sample_position = 0

# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(indata, outdata, frames, time, status):
    global write_position, sample_position

    if status:
        print(status)

    # Generate phase-continuous 19 kHz tone
    samples = np.arange(frames) + sample_position
    tone = AMPLITUDE * np.sin(2 * np.pi * FREQ * samples / FS)
    stereo = np.column_stack((tone, tone)).astype(np.float32)
    outdata[:] = stereo
    sample_position += frames

    # Store microphone channel 0
    incoming = indata[:, 0].copy()

    with lock:
        first = min(frames, BUFFER_SIZE - write_position)
        buffer[write_position:write_position + first] = incoming[:first]

        remaining = frames - first
        if remaining > 0:
            buffer[:remaining] = incoming[first:first + remaining]

        write_position = (write_position + frames) % BUFFER_SIZE

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

image = ax.imshow(
    np.full((100, 100), -100.0),
    origin="lower",
    aspect="auto",
    extent=[-HISTORY, 0, LOW_FREQ, HIGH_FREQ],
    cmap="magma",
    vmin=-100,
    vmax=-35
)

ax.axhline(FREQ, color="cyan", linestyle="--", linewidth=1, label="19 kHz carrier")
ax.set_title("Live Ultrasonic Doppler Spectrogram")
ax.set_xlabel("Time relative to now (seconds)")
ax.set_ylabel("Frequency (Hz)")
ax.legend(loc="upper right")

cbar = plt.colorbar(image, ax=ax, label="Magnitude (dB)")

status_text = ax.text(
    0.02, 0.95,
    "Gesture: NO MOTION",
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)

energy_text = ax.text(
    0.02, 0.86,
    "",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)

# -----------------------------
# Gesture decision logic
# -----------------------------
def classify_motion(freqs, mag_db):
    """
    freqs: 1D frequency axis
    mag_db: 2D spectrogram in dB with shape [freq, time]
    """

    # Focus on the most recent few columns for live detection
    recent = mag_db[:, -4:] if mag_db.shape[1] >= 4 else mag_db

    # Average over recent time columns
    mean_recent = np.mean(recent, axis=1)

    upper_mask = (freqs >= FREQ + CENTER_GUARD) & (freqs <= FREQ + SIDE_BAND)
    lower_mask = (freqs <= FREQ - CENTER_GUARD) & (freqs >= FREQ - SIDE_BAND)

    if not np.any(upper_mask) or not np.any(lower_mask):
        return "NO MOTION", -120, -120

    upper_level = np.max(mean_recent[upper_mask])
    lower_level = np.max(mean_recent[lower_mask])

    strongest = max(upper_level, lower_level)

    if strongest < MOTION_THRESHOLD_DB:
        return "NO MOTION", upper_level, lower_level

    if upper_level > lower_level + DIRECTION_MARGIN_DB:
        return "TOWARD", upper_level, lower_level
    elif lower_level > upper_level + DIRECTION_MARGIN_DB:
        return "AWAY", upper_level, lower_level
    else:
        return "MOTION / UNCERTAIN", upper_level, lower_level

# -----------------------------
# Animation update
# -----------------------------
def update(frame):
    with lock:
        current = np.concatenate((buffer[write_position:], buffer[:write_position]))

    freqs, times, spec = spectrogram(
        current,
        fs=FS,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP,
        window="hann",
        mode="magnitude"
    )

    mask = (freqs >= LOW_FREQ) & (freqs <= HIGH_FREQ)
    freqs_zoom = freqs[mask]
    mag_db = 20 * np.log10(spec[mask] + 1e-12)

    image.set_data(mag_db)
    image.set_extent([-HISTORY, 0, freqs_zoom[0], freqs_zoom[-1]])

    gesture, upper_level, lower_level = classify_motion(freqs_zoom, mag_db)

    status_text.set_text(f"Gesture: {gesture}")
    energy_text.set_text(
        f"Upper band: {upper_level:.1f} dB\nLower band: {lower_level:.1f} dB"
    )

    return image, status_text, energy_text

animation = FuncAnimation(
    fig,
    update,
    interval=120,
    blit=False,
    cache_frame_data=False
)

print("Starting live direction detector...")
print("Keep your hand still first.")
print("Then move your palm toward and away from the laptop.")
print("Close the plot window to stop.")

with sd.Stream(
    samplerate=FS,
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
    dtype="float32",
    blocksize=1024,
    callback=audio_callback
):
    plt.show()

print("Detector stopped.")