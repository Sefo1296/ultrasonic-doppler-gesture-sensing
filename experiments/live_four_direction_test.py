"""Four-direction gesture experiment with selectable HORIZONTAL / DEPTH modes.

NEW file based on the working full-range horizontal position tracker.
Neither the original horizontal nor the original TOWARD/AWAY code is changed.

Left speaker transmits 18.5 kHz; right speaker transmits 19 kHz.
Mic input is device 9, channel index 1 (the second microphone channel).

HORIZONTAL (H key): Same full-range state-tracking rules as the working
horizontal detector. Palm faces keyboard. Graph position keys M/L/R/A/D.
DEPTH (V key): Palm faces a speaker, hand moves physically closer/farther.
At the speaker zones, evaluate its OWN tone; at middle, use one or two tones
with extra confidence checks. Position is NOT changed by a depth gesture.

Mode is selected manually for this validation experiment. Doppler sign alone
cannot always distinguish a depth movement from a horizontal swipe, so this
is NOT an automatic simultaneous four-direction classifier yet.

Keys in graph: H=horizontal, V=depth, M=middle, L=left speaker,
               R=right speaker, A=outer left, D=outer right.
"""

import queue
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

# Known working hardware configuration. Do not change your other scripts.
FS = 48000
INPUT_DEVICE = 9
OUTPUT_DEVICE = 8
INPUT_CHANNELS = 4
MIC_CHANNEL = 1
AMPLITUDE = 0.02
BLOCK_SIZE = 1024
FFT_SIZE = 8192
HOP = 2048
CALIBRATION_SECONDS = 4.0
START_FRAMES = 2
STOP_FRAMES = 3
MIN_ACTIVE_FRAMES = 3
MAX_EVENT_FRAMES = 36
COOLDOWN_SECONDS = 0.5
HISTORY_SECONDS = 2.5

# Depth validation: detect meaningful, directionally consistent Doppler.
# These are initial test thresholds, not validated accuracy settings.
DEPTH_MIN_SCORE = 0.55
DEPTH_MIN_GATE_RATIO = 2.5
DEPTH_MIN_ACTIVE_FRAMES = 4
DEPTH_SINGLE_SCORE = 0.75
DEPTH_SINGLE_RATIO = 6.0
DEPTH_SINGLE_FRAMES = 6

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

ZONE_NAMES = {
    "M": "MIDDLE",
    "L": "LEFT SPEAKER",
    "R": "RIGHT SPEAKER",
    "OL": "OUTER LEFT",
    "OR": "OUTER RIGHT",
}


def decision_for(zone, signs):
    """Return (label, new_zone, transition) or None when ambiguous.

    signs = {'L': +1/-1/0, 'R': +1/-1/0}, with 0=no confident sign.

    At the speaker, the opposite speaker's Doppler can distinguish
    motion toward the middle from motion farther outward.
    """
    l, r = signs["L"], signs["R"]

    if zone == "M":
        # Conventional inner-zone pattern: L+ R- means going left,
        # L- R+ means going right. One carrier can suffice at center.
        left_votes = (l == 1) + (r == -1)
        right_votes = (l == -1) + (r == 1)
        if left_votes and not right_votes:
            return "LEFT", "L", "middle -> left speaker"
        if right_votes and not left_votes:
            return "RIGHT", "R", "middle -> right speaker"
        return None

    if zone == "L":
        # Away from left speaker: both tones down -> outer left.
        # Toward the middle: right tone up, left tone down.
        if r == -1 and l in (-1, 0):
            return "LEFT", "OL", "left speaker -> outer left"
        if r == 1 and l in (-1, 0):
            return "RIGHT", "M", "left speaker -> middle"
        return None

    if zone == "R":
        # Away from right speaker: both tones down -> outer right.
        # Toward middle: left tone up, right tone down.
        if l == -1 and r in (-1, 0):
            return "RIGHT", "OR", "right speaker -> outer right"
        if l == 1 and r in (-1, 0):
            return "LEFT", "M", "right speaker -> middle"
        return None

    if zone == "OL":
        # Outer-left -> speaker means approaching both speakers.
        if l == 1 and r == 1:
            return "RIGHT", "L", "outer left -> left speaker"
        return None

    if zone == "OR":
        # Outer-right -> speaker means approaching both speakers.
        if l == 1 and r == 1:
            return "LEFT", "R", "outer right -> right speaker"
        return None

    return None


def depth_decision_for(zone, evidence):
    """Return depth decision without altering the horizontal position.

    The nearby speaker is preferred for left/right/outer speaker zones.
    At middle require agreement if both are strong; otherwise use one
    exceptionally clear carrier, while rejecting conflicting evidence.
    """
    def strong(info, min_score, min_ratio, min_frames):
        return (
            info is not None
            and info[0] != 0
            and abs(info[1]) >= min_score
            and info[2] >= min_ratio
            and info[3] >= min_frames
        )

    def output(info, source):
        label = "TOWARD" if info[0] > 0 else "AWAY"
        return label, zone, f"depth near {ZONE_NAMES[zone]} ({source} carrier)"

    if zone in ("L", "OL", "R", "OR"):
        carrier = "L" if zone in ("L", "OL") else "R"
        info = evidence[carrier]
        if strong(info, DEPTH_MIN_SCORE, DEPTH_MIN_GATE_RATIO,
                  DEPTH_MIN_ACTIVE_FRAMES):
            return output(info, carrier)
        return None

    left, right = evidence["L"], evidence["R"]
    l_ok = strong(left, DEPTH_MIN_SCORE, DEPTH_MIN_GATE_RATIO,
                  DEPTH_MIN_ACTIVE_FRAMES)
    r_ok = strong(right, DEPTH_MIN_SCORE, DEPTH_MIN_GATE_RATIO,
                  DEPTH_MIN_ACTIVE_FRAMES)

    if l_ok and r_ok:
        if left[0] == right[0]:
            return output(left, "both")
        return None

    # One dominant carrier can still identify depth when the other
    # is absent/weak, but demand stronger evidence than for two tones.
    # Reject contradictory weaker signals with a confident sign.
    only = left if l_ok else right if r_ok else None
    other = right if l_ok else left
    source = "L" if l_ok else "R"
    if only is not None:
        if (other is not None and other[0] != 0 and
                other[0] != only[0]):
            return None
        if strong(only, DEPTH_SINGLE_SCORE, DEPTH_SINGLE_RATIO,
                  DEPTH_SINGLE_FRAMES):
            return output(only, source + " only")
    return None


class ZoneTracker:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.output_samples = 0
        self.processed_samples = 0
        self.dropped_audio_blocks = 0
        self.audio_statuses = 0

        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1 / FS)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.masks = {}
        self.histories = {}
        history_size = int(HISTORY_SECONDS * FS / HOP)
        for side, config in CARRIERS.items():
            self.masks[side] = {
                "up": self.band_mask(config["upper"]),
                "down": self.band_mask(config["lower"]),
                "view": self.band_mask(config["view"]),
            }
            self.histories[side] = deque(maxlen=history_size)

        self.calibration = {"L": [], "R": []}
        self.gates = None
        self.pending = []
        self.event = []
        self.quiet = 0
        self.cooldown_until_sample = 0
        self.display_until_sample = 0
        self.zone = "M"  # Starting position must be known.
        self.mode = "DEPTH"  # Begin by validating toward/away.
        self.label = "CALIBRATING"
        self.gate_ratios = {"L": 0.0, "R": 0.0}
        self.classified = 0
        self.uncertain = 0
        self.short_candidates = 0
        self.depth_classified = 0

    def band_mask(self, bounds):
        return (self.freqs >= bounds[0]) & (self.freqs <= bounds[1])

    def set_zone(self, zone):
        """Triggered by a graph key press; caller holds lock."""
        self.zone = zone
        self.pending.clear()
        self.event.clear()
        self.quiet = 0
        self.cooldown_until_sample = self.processed_samples + int(0.6 * FS)
        self.display_until_sample = self.processed_samples + int(1.8 * FS)
        self.label = "POSITION RESET"
        print(f"POSITION RESET -> {ZONE_NAMES[zone]}", flush=True)

    def set_mode(self, mode):
        """Change mode only on key input; clear in-progress audio event."""
        self.mode = mode
        self.pending.clear()
        self.event.clear()
        self.quiet = 0
        self.cooldown_until_sample = self.processed_samples + int(0.6 * FS)
        self.display_until_sample = self.processed_samples + int(1.5 * FS)
        self.label = f"MODE: {mode}"
        print(f"MODE -> {mode} (position: {ZONE_NAMES[self.zone]})", flush=True)

    def audio_callback(self, indata, outdata, frames, time_info, status):
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
        window_buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        filled = 0
        since_last_fft = 0
        sample_count = 0
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
            sample_count += size
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
                visuals[side] = 20 * np.log10(
                    spectrum[masks["view"]] / (FS / 2) + 1e-12
                )

            with self.lock:
                self.processed_samples = sample_count
                for side in ("L", "R"):
                    self.histories[side].append(visuals[side])
                self.process_measurement(values, sample_count)

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
            print("Calibration complete. Starting in DEPTH mode.", flush=True)
            print("DEPTH: move hand to a speaker, press L or R to set zone,")
            print("       then move palm-facing-speaker TOWARD / AWAY.")
            print("HORIZONTAL: press H, palm faces keyboard; use previous swipe rules.")
            print("Graph keys: V=depth, H=horizontal, M/L/R/A/D=set position.")
            print("Pause >=1 second between swipes.\n", flush=True)

        self.gate_ratios = {
            side: sum(values[side]) / self.gates[side] for side in ("L", "R")
        }
        if sample < self.cooldown_until_sample:
            return

        active = any(ratio > 1 for ratio in self.gate_ratios.values())
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
            threshold = config["min_score"]
            sign = 1 if score >= threshold else -1 if score <= -threshold else 0
            evidence[side] = (sign, score, peak, len(active))
        return evidence

    def finish_event(self, sample):
        frames = self.event
        self.event = []
        self.pending = []
        self.quiet = 0
        evidence = self.scores_for_event(frames)
        signs = {
            side: (evidence[side][0] if evidence[side] is not None else 0)
            for side in ("L", "R")
        }
        if self.mode == "DEPTH":
            decision = depth_decision_for(self.zone, evidence)
        else:
            decision = decision_for(self.zone, signs)
        previous = self.zone
        if decision is None:
            label = "UNCERTAIN"
            description = "state unchanged; reset position if a swipe was missed"
            self.uncertain += 1
        else:
            label, next_zone, description = decision
            self.zone = next_zone
            self.classified += 1
            if label in ("TOWARD", "AWAY"):
                self.depth_classified += 1

        details = []
        for side in ("L", "R"):
            info = evidence[side]
            if info is None:
                details.append(f"{side}=NO SIGNAL")
            else:
                _, score, peak, n = info
                details.append(f"{side}={score:+.2f} ({peak:.1f}x, {n} frames)")
        self.label = label
        self.display_until_sample = sample + int(1.5 * FS)
        self.cooldown_until_sample = sample + int(COOLDOWN_SECONDS * FS)
        print(
            f"{label:<9} | mode {self.mode} | {description} | "
            f"zone {ZONE_NAMES[previous]} -> {ZONE_NAMES[self.zone]} | "
            + " | ".join(details),
            flush=True,
        )

    def snapshot(self):
        with self.lock:
            plots = {
                side: np.column_stack(history)
                for side, history in self.histories.items()
                if history
            }
            label = self.label
            if self.gates is not None:
                if self.event:
                    label = "ANALYZING..."
                elif self.processed_samples >= self.display_until_sample:
                    label = "NO GESTURE"
            return plots, label, ZONE_NAMES[self.zone], self.mode, self.gate_ratios.copy()


def main():
    tracker = ZoneTracker()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    fig.subplots_adjust(hspace=0.3, top=0.82)
    images = {}
    for ax, side, heading in zip(
        axes, ("L", "R"),
        ("Left speaker (18.5 kHz)", "Right speaker (19 kHz)"),
    ):
        config = CARRIERS[side]
        n_bins = int(tracker.masks[side]["view"].sum())
        images[side] = ax.imshow(
            np.full((n_bins, 50), -100.0),
            origin="lower", aspect="auto",
            extent=[-HISTORY_SECONDS, 0, *config["view"]],
            cmap="magma", vmin=-100, vmax=-35,
        )
        ax.axhline(config["frequency"], color="cyan", linestyle="--", linewidth=1)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(heading)
    axes[-1].set_xlabel("Time relative to now (seconds)")

    def on_key(event):
        key = (event.key or "").lower()
        mapping = {"m": "M", "l": "L", "r": "R", "a": "OL", "d": "OR"}
        with tracker.lock:
            if key == "h":
                tracker.set_mode("HORIZONTAL")
            elif key == "v":
                tracker.set_mode("DEPTH")
            elif key in mapping:
                tracker.set_zone(mapping[key])

    fig.canvas.mpl_connect("key_press_event", on_key)

    def draw(_):
        plots, label, zone, mode, ratios = tracker.snapshot()
        for side, image in images.items():
            if side in plots:
                image.set_data(plots[side])
        fig.suptitle(
            f"FOUR-DIRECTION TEST | {mode} | {label} | Position: {zone}\n"
            f"Doppler energy / gate: Left {ratios['L']:.1f}x; Right {ratios['R']:.1f}x\n"
            "Keys: V=depth, H=horizontal, M/L/R/A/D=position",
            fontsize=11,
        )
        return tuple(images.values())

    animation = FuncAnimation(
        fig, draw, interval=180, blit=False, cache_frame_data=False
    )

    print("FOUR-DIRECTION EXPERIMENT - selectable depth / horizontal modes")
    print("Calibrate four seconds with hand stationary above keyboard middle.")
    print("Starts DEPTH mode. After calibration, move hand above LEFT speaker,")
    print("press L in the graph, then move toward/away (palm faces speaker).")
    print("To test RIGHT speaker, move hand there and press R.")
    print("To test horizontal tracking, press H (palm faces keyboard).")
    print("V switches back to depth. M/L/R/A/D reset the current zone.")
    print("Mode must be selected manually; automatic orientation detection")
    print("has NOT been implemented.\n")

    worker_thread = threading.Thread(target=tracker.worker, daemon=True)
    worker_thread.start()
    try:
        with sd.Stream(
            samplerate=FS,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            channels=(INPUT_CHANNELS, 2),
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=tracker.audio_callback,
        ):
            plt.show()
    finally:
        tracker.stop_event.set()
        worker_thread.join(timeout=2.0)
        plt.close(fig)

    print(
        f"\nFinished. Classified: {tracker.classified} "
        f"(depth: {tracker.depth_classified}); "
        f"uncertain: {tracker.uncertain}; "
        f"too-short activity: {tracker.short_candidates}; "
        f"dropped audio blocks: {tracker.dropped_audio_blocks}; "
        f"audio status warnings: {tracker.audio_statuses}"
    )


if __name__ == "__main__":
    main()
