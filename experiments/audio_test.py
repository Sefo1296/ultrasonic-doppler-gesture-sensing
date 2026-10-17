import numpy as np
import sounddevice as sd

FS = 48000
FREQ = 12000
DURATION = 2

t = np.arange(FS * DURATION) / FS

# Generate a 19 kHz tone from 0.5 to 1.5 seconds
tone = 0.02 * np.sin(2 * np.pi * FREQ * t)
tone[(t < 0.5) | (t >= 1.5)] = 0

# Send the tone to both speaker channels
audio = np.column_stack((tone, tone)).astype("float32")

print("Transmitting and recording...")

recording = sd.playrec(
    audio,
    samplerate=FS,
    channels=4,
    device=(9, 8),
    blocking=True
)

print("Recording successful!")
print("Recording shape:", recording.shape)
print("Microphone peak levels:", np.max(np.abs(recording), axis=0))

# FFT analysis of recorded audio

def measure_19khz(segment):
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    frequencies = np.fft.rfftfreq(len(segment), 1 / FS)

    band = (frequencies >= FREQ - 50) & (frequencies <= FREQ + 50)

    return 2 * np.max(spectrum[band]) / np.sum(window)


# Compare before and during transmission
before = recording[0:int(0.5 * FS), 0]
during = recording[int(0.75 * FS):int(1.25 * FS), 0]

before_level = measure_19khz(before)
during_level = measure_19khz(during)

print("19 kHz level BEFORE:", before_level)
print("19 kHz level DURING:", during_level)

if before_level > 0:
    print("Signal increase:", round(20 * np.log10(
        max(during_level, 1e-12) / before_level
    ), 2), "dB")