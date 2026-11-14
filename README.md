# Ultrasonic Doppler Gesture Sensing

**Contactless hand-gesture recognition using a laptop’s built-in speakers and microphones — no camera or external gesture sensor.**

This ongoing research prototype transmits high-frequency sound through the laptop speakers and analyzes reflections from a moving hand. Development has progressed from live Doppler visualization and manual-mode gesture detectors to a labeled audio dataset and **offline five-class machine-learning classification**.

## Latest update — 14 November 2026

- Collected **50 labeled gesture recordings** in two complete sessions (25 per session; 10 per class).
- Implemented two dataset-capture scripts, including an improved version that records the hand used on each trial and allows retakes.
- Developed a Google Colab workflow extracting **68 STFT-based Doppler features** from two microphone channels and two ultrasonic carriers.
- Evaluated **Random Forest** and **RBF Support Vector Machine (SVM)** on identical leave-one-session-out folds.
- Recorded preliminary accuracy of **98% for Random Forest (49/50)** and **94% for SVM (47/50)** on the two-session dataset.
- The trained classifiers currently operate **offline on prerecorded WAV files**. Automatic, continuous live five-class AI recognition is not yet implemented.

**Latest published weekly report:** [Random Forest and SVM Progress Report (PDF)](docs/Ultrasonic%20Weekly%20Report%20RF%20SVM%2014%202026.pdf) · [Word](docs/Ultrasonic%20Weekly%20Report%20RF%20SVM%2014%202026.docx)

**AI work:** [SVM vs Random Forest Colab notebook](Ultrasonic_SVM_vs_Random_Forest_Colab.ipynb) · [Technical AI handover](Ultrasonic_Doppler_AI_Full_Handover.md)

> **Research limitation:** The 98% and 94% scores are preliminary, session-held-out results from one participant recorded in two closely spaced sessions. They are not demonstrated real-world, cross-user or cross-device accuracy.

## Hardware and ultrasonic setup

| Component | Setup |
|---|---|
| Development laptop | HP EliteBook 845 G8 |
| Left speaker | 18,500 Hz continuous tone |
| Right speaker | 19,000 Hz continuous tone |
| Microphone recording | Four channels, 48,000 samples/second |
| Dataset recordings | 3.5 seconds per recording; 16-bit PCM WAV |
| Software | Python, NumPy, SciPy, sounddevice, Matplotlib, scikit-learn; Google Colab for AI experiments |

Earlier experiments used a single 19 kHz carrier. Dual-carrier experiments transmit both frequencies simultaneously and analyze their Doppler sidebands.

## Gesture classes and capabilities

| Capability | Current status |
|---|---|
| Ultrasonic transmission and simultaneous microphone recording | Implemented |
| Live Doppler spectrogram | Implemented |
| Toward/away rule-based detection | Demonstrated |
| Horizontal left/right and multi-zone tracking | Experimental prototypes |
| Four-direction rule-based detection | Demonstrated with **manual H/V mode selection** |
| Labeled dataset capture | Two completed 25-trial sessions |
| Offline five-class machine learning | Random Forest and RBF-SVM evaluated |
| Fully automatic real-time AI gesture recognition | **Not yet implemented** |
| Independent multi-person / different-day validation | **Not yet completed** |

The five labeled classes are **LEFT, RIGHT, TOWARD, AWAY, IDLE** (stationary). Gesture labels refer to **movement direction**, not which hand performed the motion.

## Dataset collection — 6 November 2026

Two sessions produced **50 WAV recordings**, with five examples of each class per session:

| Session | Recording count | Notes |
|---|---:|---|
| `20261106_015202_P01` | 25 | Initial capture; hand used per trial not reliably logged, therefore marked `UNKNOWN` |
| `20261106_021610_P01` | 25 | Improved capture; hand recorded per trial and bad attempts can be discarded and repeated |

The participant used a vertical hand orientation, perpendicular to the keyboard, with palm facing right. Starting positions varied. Mic channels 1 and 2 had stronger received ultrasonic signals than channels 3 and 4; the current AI features use the first two channels.

Capture programs:
- `experiments/collect_gesture_dataset.py`
- `experiments/collect_gesture_dataset_v2.py`

**Privacy and reproducibility:** Original WAV captures and the dataset ZIP remain local and are not committed to the public repository. To reproduce training, obtain the appropriately shared labeled dataset separately. Do not treat derived figures or summary metrics as substitutes for the raw recordings.

## AI signal processing and evaluation

For each complete recording, the Colab pipeline computes an STFT on each of the first two microphone channels (Hann window; 8192 samples; hop 2048 samples). It measures sideband energy **12–150 Hz above/below** each 18.5 kHz and 19.0 kHz carrier and summarizes spectral balance and activity over time. This yields **68 numerical features per recording**.

Both models use the same feature matrix, labels and **leave-one-session-out** testing: train on Session 1 and test on Session 2, then reverse. This produces 50 held-out predictions, one per recording.

| Metric | Random Forest | RBF-SVM |
|---|---:|---:|
| Accuracy | **98.00%** | **94.00%** |
| Macro precision | 98.18% | 94.85% |
| Macro recall | 98.00% | 94.00% |
| Macro F1 | 97.99% | 94.07% |
| Correct held-out predictions | 49/50 | 47/50 |

The Random Forest uses 300 trees, maximum depth 5 and minimum leaf size 2. The SVM uses an RBF kernel (`C=1.0`, `gamma="scale"`) with `StandardScaler` fitted within each training fold. Scores summarize this **small two-session pilot**, not deployment performance.

The Colab notebook also contains code to generate detailed per-class evaluation outputs; this README does not claim unshared confusion-matrix entries or identify specific misclassified gestures.

## Getting started

Clone and set up a Python environment on Windows:

```powershell
git clone https://github.com/Sefo1296/ultrasonic-doppler-gesture-sensing.git
cd ultrasonic-doppler-gesture-sensing
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Launch the original live Doppler visualizer:

```powershell
python src/live_doppler.py
```

Launch a new dataset collection session using the improved recording script:

```powershell
python experiments/collect_gesture_dataset_v2.py
```

For the offline comparison, upload [the Colab notebook](Ultrasonic_SVM_vs_Random_Forest_Colab.ipynb) to [Google Colab](https://colab.research.google.com/), then follow its cells and provide the labeled dataset ZIP when prompted. The notebook handles ZIP members with Windows-style backslash paths.

**Audio note:** Existing scripts can contain development-laptop device indices (for example, input 9/output 8), so adjust audio devices for other machines. Disabling Windows microphone enhancements was important for reception near 19 kHz. Use reasonable output volume.

## How Doppler sensing works

Reflections from a moving hand create components above and below the transmitted carriers. Their strength and evolution across time, carriers and microphone channels provide features for detecting motion and classifying gestures.

For an approximately radial reflection, the idealized frequency shift is:

```text
Δf ≈ (2 × v × f₀) / c
```

Here, `v` is radial hand speed, `f₀` is the transmitted tone and `c` is the speed of sound. For horizontal movements, multipath propagation and speaker geometry make a simple sign-only rule insufficient; the AI pipeline uses richer temporal and spectral information.

![Recorded hand-motion Doppler spectrogram](assets/Spectogram%20of%20arm%20swing%20detection.png)

## Repository layout

```text
src/           Original live Doppler and direction detectors
experiments/   Dual-carrier detection and labeled-data capture scripts
results/       Previously published analysis figures and results
assets/        Demonstration images
docs/          Main technical and weekly progress reports
captures/      Local raw audio recordings (Git-ignored)
Ultrasonic_SVM_vs_Random_Forest_Colab.ipynb
Ultrasonic_Doppler_AI_Full_Handover.md
requirements.txt
README.md
```

## Documentation and development history

- [Weekly RF/SVM report — 14 November 2026](docs/Ultrasonic%20Weekly%20Report%20RF%20SVM%2014%202026.pdf)
- [October development summary](docs/PROGRESS_UPDATE_2026-10-30.md)
- [Technical progress report v0.5 (PDF)](docs/Ultrasonic_Doppler_Gesture_Sensing_Progress_Report_v05.pdf)
- **Main technical progress report v0.7 (43 pages):** [PDF](docs/Ultrasonic%20Main%20Report%20Version%200.7%20November%202026.pdf) · [Editable Word](docs/Ultrasonic%20Doppler%20Main%20Progress%20Report%20-%20Version%200.7%20-%2014%20November%202026.docx). Covers the October development history, November labeled-data collection, and Random Forest/SVM comparison.

## Remaining work

1. Collect additional independent recording sessions with varied users, locations, speeds and days.
2. Inspect misclassifications and verify the models on fresh data before making reliability claims.
3. Test other microphone-channel combinations and improved motion segmentation.
4. Integrate the trained classifier into continuous local audio streaming, with event detection and latency testing.
5. Improve audio-device portability and reproducible reporting.

## Project team

**Saif Eldin Hany**  
**Amr Mohamed**  
**Mohamed Alaa**

Electronics and Communication Engineering Project

**Status:** Active experimental research prototype; offline AI comparison completed, continuous real-time five-class AI recognition pending.
