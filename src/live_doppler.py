
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
from threading import Lock

# Audio configuration
FS = 48000
FREQ = 19000
AMPLITUDE = 0.02

INPUT_DEVICE = 9
OUTPUT_DEVICE = 8

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 2

# Visualization configuration
HISTORY = 4
BUFFER_SIZE = FS * HISTORY
FFT_SIZE = 8192
HOP = 2048

LOW_FREQ = 18700
HIGH_FREQ = 19300

# Shared recording buffer
buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
lock = Lock()

write_position = 0
sample_position = 0


def audio_callback(indata, outdata, frames, time, status):
    global write_position, sample_position

    # Generate phase-continuous 19 kHz signal
    samples = np.arange(frames) + sample_position

    tone = AMPLITUDE * np.sin(
        2 * np.pi * FREQ * samples / FS
    )

    outdata[:] = tone[:, None]
    sample_position += frames

    # Store microphone samples
    incoming = indata[:, 0]

    with lock:
        first = min(frames, BUFFER_SIZE - write_position)

        buffer[write_position:write_position + first] = (
            incoming[:first]
        )

        remaining = frames - first

        if remaining > 0:
            buffer[:remaining] = incoming[first:]

        write_position = (
            write_position + frames
        ) % BUFFER_SIZE


# Create live spectrogram window
fig, ax = plt.subplots(figsize=(12, 6))

image = ax.imshow(
    np.full((103, 10), -100.0),
    origin="lower",
    aspect="auto",
    extent=[-HISTORY, 0, LOW_FREQ, HIGH_FREQ],
    cmap="magma",
    vmin=-100,
    vmax=-35
)

ax.axhline(
    FREQ,
    color="cyan",
    linestyle="--",
    linewidth=1,
    label="19 kHz carrier"
)

ax.set_title("Live Ultrasonic Doppler Spectrogram")
ax.set_xlabel("Time relative to now (seconds)")
ax.set_ylabel("Frequency (Hz)")
ax.legend(loc="upper right")

plt.colorbar(image, ax=ax, label="Magnitude (dB)")


def update(frame):

    # Copy the latest audio safely
    with lock:
        current = np.concatenate((
            buffer[write_position:],
            buffer[:write_position]
        ))

    # Calculate spectrogram
    frequencies, times, spectrum = spectrogram(
        current,
        fs=FS,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP,
        window="hann",
        mode="magnitude"
    )

    # Focus on frequencies around 19 kHz
    mask = (
        (frequencies >= LOW_FREQ) &
        (frequencies <= HIGH_FREQ)
    )

    magnitude_db = 20 * np.log10(
        spectrum[mask] + 1e-12
    )

    image.set_data(magnitude_db)

    return (image,)


animation = FuncAnimation(
    fig,
    update,
    interval=120,
    blit=False,
    cache_frame_data=False
)

print("Starting live Doppler spectrogram...")
print("Keep your hand still initially.")
print("Then move toward and away from the laptop.")
print("Close the graph window to stop.")

# Start simultaneous speaker output and microphone input
with sd.Stream(
    samplerate=FS,
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
    dtype="float32",
    blocksize=1024,
    callback=audio_callback
):
    plt.show()

print("Live Doppler session ended.")
