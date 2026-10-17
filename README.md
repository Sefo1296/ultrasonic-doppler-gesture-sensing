

\# Ultrasonic Doppler Gesture Sensing



\### Real-Time, Contactless Hand Gesture Recognition Using Laptop Audio Hardware



A Python-based gesture sensing system that uses a laptop's built-in speaker and microphone to detect hand motion through the acoustic Doppler effect.



The system continuously transmits a 19 kHz tone, records reflected audio, and analyzes frequency shifts to identify motion. No external sensors, cameras, or microcontrollers are required.



\## Project Overview



The objective is to develop a low-cost, contactless gesture recognition system using existing laptop hardware.



By analyzing Doppler shifts in reflected high-frequency sound, the system can identify movement toward and away from the laptop.



The project also investigates left/right gesture recognition using multichannel microphone measurements and spatial audio analysis.



\## Demonstration



!\[Arm Swing Doppler Detection](assets/Spectogram%20of%20arm%20swing%20detection.png)



\*Recorded spectrogram showing frequency changes around the 19 kHz carrier during hand movement.\*



\## Current Capabilities



| Feature | Status |

|---|---|

| 19 kHz tone generation | Implemented |

| Simultaneous audio playback and recording | Implemented |

| Live spectrogram visualization | Implemented |

| Toward/away gesture classification | Working prototype |

| No-motion detection | Working prototype |

| Multichannel microphone analysis | Experimental |

| Left/right swipe classification | Under investigation |

| Formal classification accuracy testing | Pending |



Toward/away recognition has been demonstrated experimentally. Reliable left/right recognition has not yet been established.



\## How It Works



The speaker emits a continuous high-frequency acoustic signal. When this signal reflects from a moving hand, its frequency changes according to the Doppler effect.



For an approximately radial reflection path:



\\\[

\\Delta f \\approx \\frac{2vf\_0}{c}

\\]



Where:



\- `Δf` — Doppler frequency shift (Hz)

\- `v` — radial hand velocity (m/s)

\- `f₀` — transmitted frequency (19,000 Hz)

\- `c` — speed of sound (approximately 343 m/s)



Movement toward the laptop generally produces positive frequency shifts, while movement away produces negative shifts.



The software uses short-time spectral analysis to visualize and identify these changes.



\## Technologies



\- Python 3.13

\- NumPy

\- SciPy

\- Matplotlib

\- SoundDevice

\- Windows audio interfaces (Realtek / WASAPI)



\### Development Hardware



\- HP EliteBook 845 G8

\- Built-in laptop speakers

\- Built-in microphone array

\- 48 kHz audio sampling rate

\- 19 kHz transmitted carrier



\## Installation



Clone the repository:



```bash

git clone https://github.com/Sefo1296/ultrasonic-doppler-gesture-sensing.git

cd ultrasonic-doppler-gesture-sensing

```



Create a virtual environment:



```bash

py -3.13 -m venv .venv

```



Install dependencies:



```bash

.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



The commands above assume Windows and Python 3.13.



\## Running the Project



From the repository root, start the live Doppler spectrogram:



```bash

.venv\\Scripts\\python.exe src/live\_doppler.py

```



Or start the live direction detector:



```bash

.venv\\Scripts\\python.exe src/live\_direction\_detector.py

```



\*\*Audio configuration:\*\* The scripts currently use device indices configured for the development laptop (input 9, output 8). These may need adjustment on another computer.



Before testing, disable microphone noise suppression and audio enhancements where necessary. Use a low speaker volume and avoid prolonged high-level ultrasonic output.



\## Experimental Results



\### Frequency Response



Initial tests showed severe attenuation of high-frequency signals.



After disabling Windows microphone audio enhancements, the system successfully captured the transmitted 19 kHz tone.



\### Live Doppler Detection



Live spectrograms revealed motion-related frequency components around the transmitted carrier.



Controlled hand movements produced patterns consistent with:



\- Motion toward the laptop: positive Doppler shift

\- Motion away from the laptop: negative Doppler shift

\- Stationary conditions: stable carrier with limited sideband activity



!\[Recorded Doppler Spectrogram](results/doppler\_spectrogram.png)



\### Multichannel Investigation



Four input channels were evaluated. Channels 0 and 1 provided useful Doppler measurements.



Initial left/right recognition experiments using channel-power ratios showed overlapping results, making that approach unreliable.



!\[Four-Channel Comparison](results/four\_channel\_comparison.png)



!\[Temporal Channel Analysis](results/lr\_temporal\_analysis.png)



Broadband finger-snap experiments also revealed channel-dependent arrival-time differences, suggesting potential spatial sensing capability.



\## Repository Structure



```text

src/           Main live detection applications

experiments/   Hardware testing and signal analysis

results/       Generated spectrograms and graphs

assets/        Project demonstration images

docs/          Technical reports and documentation

captures/      Local experimental audio recordings

requirements.txt

README.md

.gitignore

```



Audio recordings in `captures/` are excluded from Git tracking to avoid publishing raw microphone recordings without review.



\## Limitations



\- Detection performance depends on the laptop's audio hardware.

\- Windows audio enhancements can strongly affect high-frequency measurements.

\- Background noise and acoustic reflections may produce false detections.

\- Detector thresholds are currently calibrated experimentally.

\- Left/right gestures cannot yet be classified reliably.

\- Formal accuracy, latency, and cross-device evaluations are pending.

\- The existing implementation uses device-specific audio settings.



\## Future Development



1\. Improve motion detection stability and threshold calibration.

2\. Investigate spatial phase differences between microphone channels.

3\. Develop and validate left/right gesture recognition.

4\. Evaluate recognition accuracy with labeled experiments.

5\. Improve portability and automatic audio-device selection.

6\. Integrate recognition into a simplified real-time interface.



\## Documentation



Detailed development history, experimental measurements, figures, and limitations are available in the `docs/` directory.



\## Project Team



\*\*Saif Eldin Hany\*\*  

\*\*Amr Mohamed\*\*  

\*\*Mohamed Alaa\*\*



Electronics and Communication Engineering Project



\## Project Status



\*\*Active development — experimental prototype\*\*



The current system demonstrates real-time ultrasonic motion sensing and toward/away gesture recognition. Additional validation and gesture classification improvements remain ongoing.



