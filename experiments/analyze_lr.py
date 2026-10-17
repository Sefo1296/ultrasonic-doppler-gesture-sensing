import numpy as np
from scipy.io.wavfile import read
from scipy.signal import welch

fs, recording = read("captures/four_channels.wav")

CARRIER = 19000

def sideband_power(channel, start, end):
    signal = recording[int(start * fs):int(end * fs), channel]

    f, power = welch(
        signal,
        fs=fs,
        nperseg=8192
    )

    # Doppler frequencies, excluding the carrier
    mask = (
        ((f >= CARRIER - 200) & (f <= CARRIER - 25)) |
        ((f >= CARRIER + 25) & (f <= CARRIER + 200))
    )

    return np.sum(power[mask])

experiments = {
    "LEFT to RIGHT": (2, 4),
    "RIGHT to LEFT": (5, 7)
}

for name, (start, end) in experiments.items():

    power0 = sideband_power(0, start, end)
    power1 = sideband_power(1, start, end)

    ratio_db = 10 * np.log10(
        (power0 + 1e-20) / (power1 + 1e-20)
    )

    print(f"\n{name}")
    print(f"Channel 0 power: {power0:.9f}")
    print(f"Channel 1 power: {power1:.9f}")
    print(f"Channel 0/1 ratio: {ratio_db:.2f} dB")
