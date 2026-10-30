"""Combined ultrasonic LEFT/RIGHT test, with audio processing independent of plotting.

Test area: middle of keyboard <-> either speaker.
The two standalone, working detectors are NOT modified.

Hardware: 48 kHz, input device 9 (4 channels), output device 8 (stereo).
Left speaker = 18.5 kHz, right speaker = 19 kHz; microphone index 1.

Dependencies: numpy, sounddevice, matplotlib.
"""

import queue
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

# Same hardware and acoustic thresholds as the previous combined experiment.
FS = 48000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
INPUT_CHANNELS = 4
MIC_CHANNEL = 1
AMPLITUDE = 0.02
BLOCK_SIZE = 1024
FFT_SIZE = 8192
HOP = 2048                        # One independent measurement every 42.7 ms
CALIBRATION_SECONDS = 4.0
START_FRAMES = 2                  # Need 2 consecutive active measurements
STOP_FRAMES = 3
MIN_ACTIVE_FRAMES = 3
MAX_EVENT_FRAMES = 36             # About 1.5 seconds maximum
COOLDOWN_SECONDS = 0.50
HISTORY_SECONDS = 2.5

CARRIERS = {
    "L": {
        "frequency": 18500,
        "upper": (18535, 18680),
        "lower": (18320, 18465),
        "view": (18200, 18800),
        "median_multiplier": 6.0,
        "p90_multiplier": 2.5,
        "min_score": 0.25,
    },
    "R": {
        "frequency": 19000,
        "upper": (19035, 19180),
        "lower": (18820, 18965),
        "view": (18700, 19300),
        "median_multiplier": 12.0,
        "p90_multiplier": 4.0,
        "min_score": 0.20,
    },
}


class CombinedDetector:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.output_samples = 0
        self.dropped_audio_blocks = 0
        self.audio_statuses = 0

        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1 / FS)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.masks = {}
        self.histories = {}
        history_size = int(HISTORY_SECONDS * FS / HOP)
        for side, config in CARRIERS.items():
            self.masks[side] = {
                "up": self._band_mask(config["upper"]),
                "down": self._band_mask(config["lower"]),
                "view": self._band_mask(config["view"]),
            }
            self.histories[side] = deque(maxlen=history_size)

        self.calibration = {"L": [], "R": []}
        self.gates = None
        self.pending = []
        self.event = []
        self.quiet = 0
        self.cooldown_until_sample = 0
        self.display_until_sample = 0
        self.label = "CALIBRATING"
        self.gate_ratios = {"L": 0.0, "R": 0.0}
        self.classified = 0
        self.uncertain = 0
        self.short_candidates = 0

    def _band_mask(self, bounds):
        return (self.freqs >= bounds[0]) & (self.freqs <= bounds[1])

    def audio_callback(self, indata, outdata, frames, time_info, status):
        # No FFT or plotting here: keep the real-time callback lightweight.
        if status:
            self.audio_statuses += 1
        n = np.arange(frames) + self.output_samples
        outdata[:, 0] = AMPLITUDE * np.sin(
            2 * np.pi * CARRIERS["L"]["frequency"] * n / FS
        )
        outdata[:, 1] = AMPLITUDE * np.sin(
            2 * np.pi * CARRIERS["R"]["frequency"] * n / FS
        )
        self.output_samples += frames
        try:
            self.audio_queue.put_nowait(indata[:, MIC_CHANNEL].copy())
        except queue.Full:
            self.dropped_audio_blocks += 1

    def worker(self):
        # A rolling FFT window independent of Matplotlib redraw speed.
        window_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        filled = 0
        since_last_fft = 0
        audio_sample_count = 0
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                block = self.audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue
            size = len(block)
            if size >= FFT_SIZE:
                window_buffer[:] = block[-FFT_SIZE:]
            else:
                window_buffer[:-size] = window_buffer[size:]
                window_buffer[-size:] = block
            filled = min(FFT_SIZE, filled + size)
            audio_sample_count += size
            since_last_fft += size
            if filled < FFT_SIZE or since_last_fft < HOP:
                continue
            since_last_fft %= HOP

            spectrum = np.abs(np.fft.rfft(window_buffer * self.window))
            power = spectrum ** 2
            values = {}
            visuals = {}
            for side in ("L", "R"):
                masks = self.masks[side]
                values[side] = (
                    float(power[masks["up"]].sum()),
                    float(power[masks["down"]].sum()),
                )
                visuals[side] = (
                    20 * np.log10(spectrum[masks["view"]] / (FS / 2) + 1e-12)
                )

            with self.lock:
                for side in ("L", "R"):
                    self.histories[side].append(visuals[side])
                self.process_measurement(values, audio_sample_count)

    def process_measurement(self, values, sample):
        seconds = sample / FS
        if seconds < CALIBRATION_SECONDS:
            if seconds >= 1.0:
                for side in ("L", "R"):
                    self.calibration[side].append(sum(values[side]))
            self.label = f"CALIBRATING {CALIBRATION_SECONDS - seconds:.1f}s"
            return

        if self.gates is None:
            gates = {}
            for side, config in CARRIERS.items():
                baseline = np.asarray(self.calibration[side])
                if len(baseline) < 10:
                    self.label = "CALIBRATION FAILED - restart"
                    return
                gates[side] = max(
                    config["median_multiplier"] * np.median(baseline),
                    config["p90_multiplier"] * np.percentile(baseline, 90),
                    1e-20,
                )
            self.gates = gates
            print("Calibration complete. Begin your horizontal swipes.", flush=True)
            print("MIDDLE -> LEFT / LEFT -> MIDDLE / MIDDLE -> RIGHT / RIGHT -> MIDDLE")
            print("Pause at least one second between swipes.\n", flush=True)

        self.gate_ratios = {
            side: sum(values[side]) / self.gates[side] for side in ("L", "R")
        }
        if sample < self.cooldown_until_sample:
            return

        active = any(self.gate_ratios[side] > 1 for side in ("L", "R"))

        if self.event:
            if active:
                self.event.append(values)
                self.quiet = 0
            else:
                self.quiet += 1
            if self.quiet >= STOP_FRAMES or len(self.event) >= MAX_EVENT_FRAMES:
                self.finish_event(sample)
        elif active:
            self.pending.append(values)
            if len(self.pending) >= START_FRAMES:
                self.event = self.pending[:]
                self.pending = []
                self.quiet = 0
        else:
            if self.pending:
                self.short_candidates += 1
            self.pending = []

    def scores_for_event(self, frames):
        evidence = {}
        for side, config in CARRIERS.items():
            active = [
                frame[side] for frame in frames
                if sum(frame[side]) > self.gates[side]
            ]
            if len(active) < MIN_ACTIVE_FRAMES:
                evidence[side] = None
                continue
            keep = min(len(active), max(3, int(np.ceil(len(active) * 0.4))))
            strongest = sorted(active, key=sum, reverse=True)[:keep]
            up = sum(v[0] for v in strongest)
            down = sum(v[1] for v in strongest)
            score = (up - down) / (up + down + 1e-30)
            peak = max(sum(v) for v in active) / self.gates[side]
            if abs(score) < config["min_score"]:
                direction = None
            elif side == "L":
                direction = "LEFT" if score > 0 else "RIGHT"
            else:
                direction = "RIGHT" if score > 0 else "LEFT"
            evidence[side] = (direction, score, peak, len(active))
        return evidence

    def finish_event(self, sample):
        frames = self.event
        self.event = []
        self.pending = []
        self.quiet = 0
        evidence = self.scores_for_event(frames)
        directions = [
            result[0] for result in evidence.values()
            if result is not None and result[0] is not None
        ]
        if directions and len(set(directions)) == 1:
            label = directions[0]
            self.classified += 1
        else:
            label = "UNCERTAIN"
            self.uncertain += 1

        descriptions = []
        for side in ("L", "R"):
            result = evidence[side]
            if result is None:
                descriptions.append(f"{side}=NO SIGNAL")
            else:
                direction, score, peak, n = result
                descriptions.append(
                    f"{side}={score:+.2f} ({peak:.1f}x, {n} frames, "
                    f"{direction or 'weak direction'})"
                )
        self.label = label
        self.display_until_sample = sample + int(1.5 * FS)
        self.cooldown_until_sample = sample + int(COOLDOWN_SECONDS * FS)
        print(f"{label:<10} | " + " | ".join(descriptions), flush=True)

    def snapshot(self):
        with self.lock:
            plots = {}
            for side in ("L", "R"):
                if self.histories[side]:
                    plots[side] = np.column_stack(self.histories[side])
            label = self.label
            processed_samples = getattr(self, "processed_samples", None)
            if self.gates is not None:
                if self.event:
                    label = "ANALYZING..."
                elif self.gate_ratios and self.display_until_sample < self.output_samples:
                    label = "NO GESTURE"
            ratios = self.gate_ratios.copy()
            return plots, label, ratios


def main():
    detector = CombinedDetector()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    fig.subplots_adjust(hspace=0.30, top=0.85)
    images = {}
    for ax, side, heading in zip(
        axes,
        ("L", "R"),
        ("Left speaker (18.5 kHz)", "Right speaker (19 kHz)"),
    ):
        config = CARRIERS[side]
        n_bins = detector.masks[side]["view"].sum()
        images[side] = ax.imshow(
            np.full((int(n_bins), 50), -100.0),
            origin="lower",
            aspect="auto",
            extent=[-HISTORY_SECONDS, 0, *config["view"]],
            cmap="magma", vmin=-100, vmax=-35,
        )
        ax.axhline(config["frequency"], color="cyan", linestyle="--", linewidth=1)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(heading)
    axes[-1].set_xlabel("Time relative to now (seconds)")

    def draw(_):
        plots, label, ratios = detector.snapshot()
        for side, image in images.items():
            if side in plots:
                image.set_data(plots[side])
        fig.suptitle(
            f"COMBINED LIVE LEFT/RIGHT | {label}\n"
            f"Doppler energy / noise gate: Left {ratios['L']:.1f}x; "
            f"Right {ratios['R']:.1f}x",
            fontsize=12,
        )
        return tuple(images.values())

    animation = FuncAnimation(fig, draw, interval=180,
                              blit=False, cache_frame_data=False)

    print("COMBINED LEFT/RIGHT EXPERIMENT - independent audio processing")
    print("Keep your hand still for the first four seconds.")
    print("Use only the middle <-> left and middle <-> right swipe zones.")
    print("Close any other Doppler programs before running.\n")

    worker_thread = threading.Thread(target=detector.worker, daemon=True)
    worker_thread.start()
    try:
        with sd.Stream(
            samplerate=FS,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            channels=(INPUT_CHANNELS, 2),
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=detector.audio_callback,
        ):
            plt.show()
    finally:
        detector.stop_event.set()
        worker_thread.join(timeout=2.0)
        plt.close(fig)

    print(
        f"\nFinished. Classified: {detector.classified}; "
        f"uncertain: {detector.uncertain}; "
        f"too-short activity: {detector.short_candidates}; "
        f"dropped audio blocks: {detector.dropped_audio_blocks}; "
        f"audio status warnings: {detector.audio_statuses}"
    )


if __name__ == "__main__":
    main()
