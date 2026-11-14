# Full Project Handover — Ultrasonic Doppler Gesture Recognition Using Random Forest and SVM

**Status:** Preliminary, offline, five-class gesture classification pipeline implemented in Google Colab. Two algorithms compared with the same features and session-based test folds. This handover is for a teammate or another ChatGPT conversation to develop a mini technical report. **Do not interpret the current scores as validated performance on new people or settings.**

## 1. Project purpose and hardware

We are building a camera-free hand-gesture recognition system using the **ultrasonic Doppler effect**. The device is an **HP EliteBook 845 G8** laptop. It simultaneously transmits two tones:

- Left laptop speaker: **18,500 Hz (18.5 kHz)**.
- Right laptop speaker: **19,000 Hz (19.0 kHz)**.
- Recording: laptop microphones, **four audio channels**, sampled at **48,000 samples/s**.
- Original files: **16-bit PCM WAV**; each recording lasts **approximately 3.5 s**.
- The moving hand reflects the transmitted sound, producing changes in energy above and below the carrier frequencies (Doppler sidebands).
- Objective: predict exactly one of **LEFT, RIGHT, TOWARD, AWAY, IDLE** from each WAV recording. The gesture label indicates **movement direction**, not left/right hand identity.

The goal at this stage is **offline classification** of complete prerecorded gestures. Live microphone inference and continuous real-time recognition have **not** been implemented in this work.

## 2. Gesture collection and data description

We have **50 labeled WAVs total**, from **one participant**, split into two sessions recorded close in time:

| Session | Folder | Recordings | Gesture count | Notes |
|---|---|---:|---|---|
| 1 — initial pilot | `20261106_015202_P01` | 25 | 5 each of the five classes | Both hands were used, but hand identity was not reliably logged. Its CSV `hand` value was corrected to `UNKNOWN`. |
| 2 — improved capture | `20261106_021610_P01` | 25 | 5 each of the five classes | Improved capture software; left/right hand recorded for each trial; incorrect attempts discarded and retaken. |
| **Total** | | **50** | **10 per class** | Only one participant and two closely related sessions. |

- Each folder includes a `labels.csv` with the label and WAV filename. ZIP listings also showed `labels_original.csv` and `session.json` in at least the first session. The pipeline uses **`labels.csv` as the source of ground truth** and does not rely on filename text as a feature.
- Subject held the palm **vertical, perpendicular to the keyboard, facing RIGHT** during recordings.
- Horizontal movements included middle-to-speaker and speaker-to-middle trajectories; depth gestures occurred near either laptop speaker.
- Requested speed: **NORMAL**, but speed was not measured quantitatively.
- Hand identity and starting position differed between trials; these factors may affect Doppler signatures.
- The original audio and label files are preserved; notebook extraction and results are separate working copies.

Actual dataset archive uploaded in Colab: **`gesture_dataset_50_samples.zip`**, around **48.5 MB**. The Colab check detected **50 WAV files**. It initially reported zero labels only because its ZIP path handling was incorrect (see §4).

## 3. Early signal observations (before machine learning)

Exploratory measurements confirmed both ultrasonic carrier tones could be found in the recorded microphone signals. Microphone **channels 1 and 2** carried stronger recorded ultrasonic energy than channels 3 and 4, which is why the initial AI pipeline uses channels 1–2 only.

Moving-hand recordings generally showed more Doppler sideband energy than the IDLE class. The *preliminary average* above/below-carrier energy patterns from Session 2 were:

| Ground-truth gesture | Near 18.5 kHz | Near 19.0 kHz |
|---|---|---|
| LEFT | Positive | Negative |
| RIGHT | Negative | Positive |
| TOWARD | Positive | Positive |
| AWAY | Negative | Negative |

Here Positive/Negative refer to the dominant relative sideband direction, **not** model predictions or definitive physical rules. In the first analyzed microphone, expected signs were seen in all five examples of RIGHT, TOWARD, and AWAY, but in only **one of five LEFT** examples. The cause of LEFT variability was not established; hand identity and movement position were possible contributors. Consequently, we did **not** implement gesture labels using a brittle sign-only heuristic; we trained classifiers from the CSV labels.

## 4. Google Colab setup and ZIP extraction problem

We created successive `.ipynb` notebooks for Google Colab: initial general pipeline; Random Forest version; upload-button version; Windows ZIP-fixed Random Forest version; and **SVM-vs-Random-Forest comparison notebook**.

To start: open **https://colab.research.google.com**, choose **File → Upload notebook**, select the `.ipynb`, connect the runtime, and run the code cells in order. In the first code cell, `google.colab.files.upload()` opens a **Choose Files** picker. Upload the unchanged dataset ZIP; no manual Windows or Linux path is needed.

**The actual bug:** The Windows ZIP stored member names such as `20261106_015202_P01\\labels.csv` (backslashes). A Colab/Linux snippet checked `Path(member_name).name == 'labels.csv'`, which treated the backslash as an ordinary character and wrongly counted **0 CSVs**, although listing the ZIP displayed `labels.csv`. WAVs were correctly counted (50).

**Fix:** Before extracting entries, replace `\\` with `/`, build safe relative paths, and extract into `/content/ultrasonic_dataset`. The corrected code checks against unsafe paths and locates WAVs and CSVs recursively. Expected detection: **50 WAVs, 2 `labels.csv` files**. An optional `labels_original.csv` is not counted as the active labels file. No re-recording or manual re-zipping is required.

The latest **SVM-vs-Random-Forest Colab notebook** already incorporates this ZIP fix.

## 5. Data parsing and integrity checks

The code uses Python, `pandas`, NumPy, SciPy, and scikit-learn. It:

1. Loads both `labels.csv` files with UTF-8 BOM support.
2. Identifies the gesture-label column and WAV-name/path column even if standard column names vary.
3. Normalizes harmless gesture aliases (e.g., `TOWARDS` → `TOWARD`, `STILL` → `IDLE`).
4. Matches each CSV row to its original WAV **within the correct session**; checks for repeated WAV matching.
5. Builds metadata with `path`, `gesture`, `session` and source `csv`.
6. Shows gesture counts per session; verifies there are two groups; reads a sample recording per session to check **48 kHz** and at least **two channels**.

**Never** use WAV filename tokens, class labels, session ID, or hand metadata as input predictors. The class label (`y`) is used only as the training/testing target; the session is used only to define holdout folds.

## 6. Signal preprocessing and feature extraction — identical for both models

Each complete WAV file is reduced to a fixed-length **68-element numerical feature vector**. Only microphone **channels 1 and 2** are used (zero-based `CHANNELS = (0, 1)`); channels 3 and 4 are not yet evaluated as inputs.

- Read WAV with `scipy.io.wavfile.read`.
- Convert integer audio to normalized `float64` amplitude (so the raw PCM range does not vary by datatype).
- Compute **STFT (short-time Fourier transform)** for each selected microphone, using SciPy `stft` with Hann window, `nperseg=8192`, `noverlap=6144` and `hop_size=2048`; `boundary=None`, `padded=False`.
- Sampling rate: **48,000 Hz**. The carrier frequencies are **18,500 Hz** and **19,000 Hz**.
- From each STFT, form **power spectrogram** (`abs(Z)**2`). For each carrier, sum power in an **upper sideband from +12 to +150 Hz**, a **lower sideband from −150 to −12 Hz**, and a **central/carrier band within ±12 Hz**.
- Construct log-ratio time series (with a small epsilon to avoid log(0)):
  - `balance = log10((positive + eps)/(negative + eps))`
  - `activity = log10((positive + negative + eps)/(center + eps))`
  - `positive_activity = log10((positive + eps)/(center + eps))`
  - `negative_activity = log10((negative + eps)/(center + eps))`
- For `balance`, take **mean, standard deviation, median, 10th percentile, 90th percentile** (5 features).
- For `activity`, take the same **5 summary statistics**.
- Add **90th percentile and median** of `positive_activity` and `negative_activity` (4 features).
- Add **fraction of STFT frames with positive balance** (1 feature).
- Add **average balance in first and second half of recording** (2 features).
- Thus **17 features per microphone/carrier pair × 2 microphones × 2 carriers = 68 features per WAV**.
- Assemble `X_df` (50 × 68), labels `y` (50 entries), and `groups` (session identifier for each recording). Check all features are finite.

The exact feature engineering has been verified from the notebooks. These 68 hand-engineered spectral features—not raw spectrogram-image CNN inputs—were used for both Random Forest and SVM.

## 7. Evaluation design — crucial to explain accurately

**Leave-One-Group-Out by recording session** (`sklearn.model_selection.LeaveOneGroupOut`) was used, with two folds:

- **Fold A:** train on all 25 recordings from Session 1; test on all 25 recordings from Session 2.
- **Fold B:** train on all 25 recordings from Session 2; test on all 25 recordings from Session 1.

Every WAV is predicted **once by a model that was not trained on its entire recording session**. Predictions from the two 25-recording test folds are combined to calculate the reported scores across 50 held-out predictions.

This is **not** a conventional random 80/20 split, and we do **not** split overlapping windows from the same recording between train and test. Feature extraction itself uses a fixed formula, while SVM's **StandardScaler is fitted only within each training fold** via a scikit-learn pipeline.

The same 50 WAVs, 68 features, and fold indices were used for both models, making the head-to-head comparison fair **within this small dataset**. There are only two separate session groups, both from the same person close in time; leave-one-session-out is useful, but **not an independent new-subject/new-day validation**.

## 8. Model 1 — Random Forest

Library: `sklearn.ensemble.RandomForestClassifier`. Fixed model configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

A forest of 300 decision trees models nonlinear interactions among handcrafted Doppler features. Standard scaling is not required for tree splits. This model was first developed in the Random Forest notebook and then **rerun from scratch** in the combined SVM-vs-RF notebook on the **identical folds**.

## 9. Model 2 — Support Vector Machine (SVM)

Chosen because RBF-SVM is a reasonable second benchmark for a **small, numerical feature dataset**. Libraries: `sklearn.svm.SVC`, `sklearn.preprocessing.StandardScaler`, `sklearn.pipeline.make_pipeline`.

```python
svm_pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
)
```

The RBF kernel models nonlinear class separation. Scaling is important because SVM is sensitive to feature magnitudes. The scaler is inside the pipeline so it learns statistics only from each training fold. Hyperparameters were **fixed**, not selected through an independently nested tuning procedure. The code did **not** enable calibrated probability output (`probability=True`).

## 10. CONFIRMED RESULTS from the user's Colab comparison screenshot

The screenshot title was **`=== SVM vs RANDOM FOREST (SAME SESSION SPLITS) ===`** and showed:

| Metric | Random Forest | SVM (RBF) |
|---|---:|---:|
| Accuracy | **98.00%** | **94.00%** |
| Macro precision | **98.18%** | **94.85%** |
| Macro recall | **98.00%** | **94.00%** |
| Macro F1-score | **97.99%** | **94.07%** |
| Correct / 50 held-out predictions | **49 / 50** | **47 / 50** |
| Incorrect / 50 | **1** | **3** |

RF exceeded SVM by **4 percentage points** in accuracy on these combined held-out-session predictions; that is a difference of only **two recordings**. There were five balanced classes; naive uniform chance performance would be 20% accuracy. The strong scores are interesting but do **not** demonstrate generalization to new participants, hardware, days, room geometries, hand positions, or velocities.

**Metrics:** accuracy = correct predictions / total; macro precision = unweighted mean of the five class-wise precisions; macro recall = unweighted mean of class-wise recalls; macro F1 = unweighted mean of class-wise F1 scores. A confusion matrix is 5×5 (rows = ground-truth class; columns = predicted class); correct predictions lie on its diagonal.

**Not available in this conversation:** the actual 5×5 confusion-matrix numbers/images, per-gesture classification reports, per-fold accuracies, specific filenames misclassified, and exact feature-importance values. The notebooks produce these outputs, but the user only shared the combined comparison table. **Do not invent or infer any particular misclassified gesture.**

**Interpretation:** The preliminary data favors RF, but the score is from a very limited, closely related two-session dataset. The small test set makes percentage estimates volatile: one prediction changes overall accuracy by **2 percentage points**. Even 98% does not automatically imply leakage, but it justifies checking for leakage and collecting independent tests.

## 11. Notebook cell sequence and output files

### Corrected Random Forest notebook — 6 code cells
1. **ZIP upload and safe extraction**, including Windows backslash fix.
2. **CSV/WAV parsing and integrity checks**.
3. **68 STFT-based features** from two microphones around both carrier frequencies.
4. **Random Forest leave-one-session-out training/evaluation**.
5. **Accuracy, macro precision, macro recall, macro F1, classification report, confusion matrix**, CSV predictions including `wav`, `session`, `actual`, `predicted`, `correct`.
6. **Fit final RF on all 50 WAVs**, save joblib model bundle, and download `ultrasonic_random_forest_results.zip` containing:
   - `random_forest_gesture_model.joblib`
   - `random_forest_confusion_matrix.png`
   - `random_forest_classification_report.csv`
   - `random_forest_predictions.csv`
   - `random_forest_fold_scores.csv`

### SVM-vs-RF comparison notebook — 7 code cells
1. ZIP upload/extract (Windows path fix).
2. Data and labels checks.
3. Shared 68-feature Doppler STFT extraction.
4. SVM fitting and leave-one-session-out prediction.
5. SVM metrics, per-class report, confusion matrix, list of incorrect WAVs.
6. **Train and score RF on the exact same features and folds**; show comparison table; RF confusion matrix; comparison of per-WAV predictions/disagreements.
7. **Train final SVM on all 50 WAVs** (for later use, **not** the source of held-out accuracy), save a joblib bundle, and download `ultrasonic_svm_vs_random_forest_results.zip`, which includes:
   - `svm_gesture_model.joblib`
   - `svm_confusion_matrix.png`
   - `svm_classification_report.csv`
   - `svm_predictions.csv`
   - `svm_fold_scores.csv`
   - `svm_vs_random_forest_comparison.csv`
   - `random_forest_same_split_confusion_matrix.png`
   - `random_forest_fold_scores_same_split.csv`
   - `svm_vs_rf_per_recording.csv`

Both final saved `.joblib` bundles store model and feature configuration/column order. For prospective inference, extract the **exact same 68 features in the same order** as training; SVM scaling is stored inside its trained pipeline. The saved final models have been fitted on all 50 samples, while reported accuracy comes from separate held-out fold models. **Do not evaluate the final all-data model on those same 50 samples and call it test accuracy.**

## 12. Analyses discussed but NOT confirmed completed

These were **suggested next additions**, not reported empirical results:

- **RF feature-importance bar plot:** use `final_random_forest.feature_importances_`, paired with `X_df.columns`, to show the 15 most influential features. Impurity-based importance does not prove causation and can be biased for correlated features.
- **Misclassification analysis:** filter the RF `prediction_table` where `correct == False`; inspect `wav`, `session`, `actual`, `predicted`. Do the same for `svm_prediction_table` and inspect the paired predictions CSV.
- **Spectrogram plots:** compare STFT near both carriers for gesture classes and IDLE; visualize positive/negative Doppler sidebands.
- **Test microphones 3 and 4:** compare channel combinations and evaluate whether extra channels improve robustness. Current confirmed models use only microphones 1–2.
- **Independent validation:** collect a third session on another day (suggested: ~50 new trials, 10/class), freeze the model/feature choices before testing, and ideally recruit more people, vary hand/distance/location/speed and environment.
- **Hyperparameter tuning:** only with appropriate nested training-only validation; do not optimize using the held-out evaluation sessions and then claim those same sessions as unbiased test data.
- **New WAV inference:** load `joblib`, compute the same features, predict a class; no completed example with unseen WAV has been provided.
- **Real-time inference:** stream laptop microphones, apply processing on controlled time windows, recognize gestures continuously. Not implemented. Google Colab itself is mainly for offline training, not reliable low-latency local microphone streaming.
- **Other possible classifier:** CNN using spectrograms only after collecting enough independent data; with 50 recordings, simple engineered-feature baselines are more defensible.

## 13. Critical limitations and reporting rules

1. **One participant, two closely spaced sessions:** no evidence of generalization across people/days/environments.
2. **50 total WAVs (25 per session):** estimates are statistically imprecise; just a couple of errors change percentages substantially.
3. **Hand identity not reliably logged for Session 1; positions and hands varied.** The LEFT Doppler sign was inconsistent in exploratory analysis.
4. **No actual velocity measurements.** `NORMAL` was a requested speed label, not a measured speed.
5. **Only channels 1 and 2 used** in the 68-feature AI pipeline; four channels were recorded.
6. **No fully independent prospective test set.** Both models were compared only by leave-one-session-out evaluation. This is not deployment validation.
7. **No formal test for leakage** has been demonstrated in the shared conversation beyond the pipeline’s session grouping and training-only SVM scaler. Dataset/metadata checks and independent tests remain valuable.
8. **Comparing accuracy 98% versus 94% does not establish RF is universally superior.** The difference corresponds to two samples and must be presented as preliminary.
9. **The specific confusion-matrix entries and class-wise errors have not been supplied** to this chat; request actual saved images/CSVs before claiming which gesture is hardest for the classifiers.
10. **No real-time results, model latency/CPU measurements, confidence calibration, or multi-subject performance** have been measured.
11. **Do not modify original WAV and `labels.csv` files.** All processing should be in separate output directories.

## 14. Suggested mini-report organization

1. **Title:** *Ultrasonic Doppler-Based Hand Gesture Recognition Using Random Forest and RBF-SVM: A Preliminary Study*.
2. **Abstract:** task, setup, dataset size, 68-feature STFT processing, two models, leave-one-session-out results, and limitations.
3. **Introduction / Objective:** touchless vision-free five-class gestures using laptop ultrasonic reflections.
4. **Experimental Setup:** HP EliteBook 845 G8, 18.5/19.0 kHz tones, four-channel 48 kHz PCM recording, 3.5s WAVs, hand orientation.
5. **Dataset:** two session IDs, 25 recordings each, 10 per class total, label details and limitations.
6. **Signal Processing / Feature Engineering:** STFT, sidebands ±12–150 Hz, positive/negative/center energy ratios, 68 features from two channels. Diagram: sound → microphone → STFT → features → classifier → one of five labels.
7. **ML Methods:** RF and RBF SVM with exact hyperparameters, and scaling inside SVM.
8. **Validation / Metrics:** session-based 2-fold LOGO, why recording/window leakage matters, accuracy, macro precision/recall/F1, confusion matrices.
9. **Results:** exact table above and **actual** confusion-matrix images to be inserted from Colab (do not invent entries).
10. **Discussion:** RF higher in observed accuracy; only 1 vs 3 errors; small data, possible session/hand-position dependency, note exploratory LEFT inconsistency as a hypothesis rather than verified class error.
11. **Limitations / Future Work:** new sessions/participants, more microphone channels, visualizations, independent testing, then live inference.
12. **Conclusion:** both methods performed well on this controlled dataset, with RF ahead in this preliminary comparison; further generalization studies required.

### Suggested ready-to-use results paragraph

> Using 50 labeled ultrasonic WAV recordings across two collection sessions, we extracted 68 STFT-based Doppler features from two microphone channels around 18.5 and 19.0 kHz. Random Forest and RBF-SVM classifiers were evaluated using leave-one-session-out cross-validation, with identical features and held-out folds. The Random Forest achieved 98.00% accuracy, 98.18% macro precision, 98.00% macro recall, and 97.99% macro F1-score, whereas the RBF-SVM achieved 94.00% accuracy, 94.85% macro precision, 94.00% macro recall, and 94.07% macro F1-score. These correspond to 49 and 47 correctly classified held-out recordings, respectively. While Random Forest performed better within this dataset, the results are preliminary because the recordings originated from one participant over two closely spaced sessions. Evaluation on additional independent recordings and participants is necessary before claiming generalizable gesture recognition.

## 15. Information/files the teammate should request for the final report

- The actual `gesture_dataset_50_samples.zip` **only if permitted to share** (contains the original WAV files and labels). For reporting, the notebook/result exports may suffice without sharing raw recordings.
- The current Colab `.ipynb` notebook, preferably **`Ultrasonic_SVM_vs_Random_Forest_Colab.ipynb`**, which compares both models under the same evaluation protocol.
- The downloaded `ultrasonic_svm_vs_random_forest_results.zip` and/or `ultrasonic_random_forest_results.zip`, especially **both confusion matrix PNGs**, `svm_vs_rf_per_recording.csv`, per-class reports and per-fold scores.
- Any true spectrogram or signal-analysis figures previously created outside the notebook (they were discussed, but their images/underlying numbers are not in this conversation).
- Instructor-required report length, author names/IDs, citation style, and figures if applicable (not specified in this thread).

**Important for the next chat:** Work only from the evidence supplied; keep confirmed observations separate from suggestions. Never fabricate confusion matrices, error-file names, hyperparameter search results, third-session accuracy, or real-time testing performance.
