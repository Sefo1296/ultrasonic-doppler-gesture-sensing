import numpy as np
import sounddevice as sd

FS = 48000
frequencies = [1000, 8000, 10000, 12000, 16000, 19000]

print("Starting frequency scan...\n")

for freq in frequencies:
    t = np.arange(FS * 2) / FS

    tone = 0.02 * np.sin(2 * np.pi * freq * t)
    tone[(t < 0.5) | (t >= 1.5)] = 0

    audio = np.column_stack((tone, tone)).astype("float32")

    recording = sd.playrec(
        audio,
        samplerate=FS,
        channels=4,
        device=(9, 8),
        blocking=True
    )

    signal = recording[int(0.75*FS):int(1.25*FS), 0]

    window = np.hanning(len(signal))
    spectrum = 2 * np.abs(np.fft.rfft(signal * window)) / sum(window)
    fft_freqs = np.fft.rfftfreq(len(signal), 1/FS)

    band = abs(fft_freqs - freq) <= 25
    amplitude = np.max(spectrum[band])

    print(f"{freq:5d} Hz | Signal amplitude: {amplitude:.9f}")

print("\nScan completed.")