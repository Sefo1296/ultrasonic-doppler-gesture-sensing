# Progress update — 30 October 2026

This update describes **interactive engineering prototype tests**, not a held-out recognition-accuracy benchmark.

## Recorded milestones

- **15–16 October:** full-duplex 48 kHz sound acquisition; 19 kHz live spectrogram and original toward/away indicator; Windows microphone audio enhancement interference identified and disabled; initial left/right channel-power comparisons were not reliable.
- **17 October:** initial public GitHub repository established with source, experimental scripts, figures, and progress report v0.5.
- **29 October:** confirmed stereo routing (output channel 0 = physical left speaker; channel 1 = right speaker), transmitted independent **18.5 kHz left / 19 kHz right** carriers, collected dual-tone baseline and swipe recordings.
- **29–30 October:** changed from whole-recording averages to short-event sideband scores; built separate left-side and right-side detectors. Moving real-time analysis independently of Matplotlib refresh improved gesture responsiveness.
- **30 October:** implemented five-zone horizontal position tracking with manual reset; integrated LEFT, RIGHT, TOWARD and AWAY labels into one app with **manual H (horizontal) / V (depth) selection**.

## Current system

| Function | Evidence and status |
|---|---|
| Original 19 kHz toward/away | Demonstrated in `src/live_direction_detector.py` |
| Stereo dual-carrier output | 18.5 kHz left; 19 kHz right |
| Sideways movement | Interactive tracking across outer-left ↔ left speaker ↔ middle ↔ right speaker ↔ outer-right |
| Five-zone state | Estimated from successive gestures; operator resets after a missed transition |
| Toward/away near a speaker | Depth-mode experimental test on both physical speaker positions |
| Four direction labels | Available in one experimental app; mode must be manually selected |
| Automatic depth/horizontal separation | **Not yet implemented** |
| Independent recognition benchmark | **Not yet performed** |

### Console-observed events

- Horizontal full-range tracker demonstration: **55 classified events, 36 uncertain events**, zero dropped audio blocks. Many uncertain events appeared during explorative movement and mistaken/reset zone state.
- Combined manually selected four-direction program (one test): **52 classified events, including 22 depth events; 10 uncertain events**, zero dropped audio blocks.
- Depth-only run over both speakers: **53 classified depth events, 5 uncertain events**, zero dropped audio blocks.

**These are software event counts, not per-gesture accuracy rates.** They do not provide the number of attempted gestures, misses, or false positives over measured idle time.

## Configuration and gesture conventions

The development laptop was an HP EliteBook 845 G8 using Realtek WASAPI input device 9 (4 channels) and output device 8 (2 channels), 48 kHz sample rate. Later prototype tests primarily used microphone channel index 1; the initial toward/away detector used channel 0. Device indices are laptop-specific.

- **Horizontal:** palm faces the keyboard; lateral motion between adjacent tracked zones.
- **Depth:** palm faces the speaker; hand moves toward or away from the source.
- **Position reset:** in the tracker, M = middle, L = left speaker, R = right speaker, A = outer left, D = outer right.
- **Mode selection:** H = horizontal; V = depth. Automatic gesture-family/orientation identification remains future work.

## Next engineering tasks

1. Collect fixed-count, manually labeled test trials for LEFT, RIGHT, TOWARD, and AWAY, including missed, incorrect, and uncertain events.
2. Distinguish depth from lateral movement without pressing H or V, with explicit rejection when ambiguous.
3. Reduce manual reset dependence after missed movements.
4. Measure latency, different hand speeds, off-axis geometry and idle false-positive rate.
5. Synchronize the latest reviewed `experiments/` scripts and the updated v0.6 Word/PDF report to GitHub.

The original detailed v0.5 report remains at [docs/Ultrasonic_Doppler_Gesture_Sensing_Progress_Report_v05.pdf](Ultrasonic_Doppler_Gesture_Sensing_Progress_Report_v05.pdf). The expanded v0.6 report was prepared on 30 October and **must be uploaded with the latest code** before its GitHub PDF link is advertised.
