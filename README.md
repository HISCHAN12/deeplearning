# TFT Recording Analysis and Deep Learning Advisor

## Student

- Student ID: 22000708
- Name: Heechan Jung
- Course: Deep Learning Term Project

## Project Title

**A Deep Learning Framework for TFT Recording Analysis, Placement Prediction, and Strategic Action Recommendation**

## Project Links

- GitHub Pages: `https://hischan12.github.io/deeplearning/`
- YouTube demo: `https://youtu.be/YOUR_VIDEO_ID`

The YouTube placeholder must be replaced after the approximately seven-minute presentation video is uploaded.

## Introduction

Teamfight Tactics combines champions, traits, items, augments, economy, leveling, and spatial positioning. These variables interact non-linearly, making TFT a useful prototype domain for representation learning, spatial feature extraction, classification, clustering, and recommendation.

This repository provides two related course prototypes:

1. **Offline recording analysis:** samples a completed TFT recording from beginning to end and produces a timeline of predicted placement, Top 4 probability, and visual-proxy action recommendations.
2. **Manual advisor demo:** accepts manually selected composition, board style, and game-flow values to demonstrate placement prediction and cluster-based action recommendation.

The automatic recording analyzer is the preferred policy-safe demonstration. It does not analyze a live match. Its offline action suggestions are reference estimates rather than live recommendations based on recognized HUD data.

## Current Capabilities

### Offline Recording Analyzer

The analyzer processes a selected video continuously at a configurable interval.

```text
0 seconds  -> frame analysis -> placement and Top 4 prediction
5 seconds  -> frame analysis -> placement and Top 4 prediction
10 seconds -> frame analysis -> placement and Top 4 prediction
...
video end  -> JSON and HTML timeline report
```

It automatically:

- Opens MP4, MKV, AVI, MOV, or WEBM recordings.
- Samples frames every N seconds.
- Extracts the center board region.
- Divides the visual board into a 4x7 grid.
- Measures saturation, brightness, Canny edge density, occupied-cell count, and frame-to-frame activity.
- Produces 1st-to-8th placement probabilities and Top 4 probability for each sampled timestamp.
- Produces a reference action recommendation from estimated visual board strength and its change over time.
- Saves a JSON result and an HTML probability graph.

The full report is shown after analysis completes. Predictions are calculated throughout the recording, but they are not streamed during gameplay.

### Manual Advisor and Overlay

The repository also contains:

- A local browser advisor.
- A Windows always-on-top manual-input overlay.
- Manual game-state snapshots for stage, health, gold, level, streak, board strength, and unspent items.
- Short-term flow tracking using changes between the previous and current submitted states.
- Cluster-based reference actions such as level up, roll down, hold economy, slam item, and reposition carry.

These modes are prototypes and do not automatically read the live TFT board.

## Deep Learning and Algorithms

### 1. Multi-hot Encoding

Champion, trait, item, and augment selections are converted into a high-dimensional deck vector.

### 2. Autoencoder

A NumPy autoencoder compresses the sparse deck vector into an eight-dimensional latent embedding. Reconstruction mean-squared error is minimized through backpropagation.

### 3. K-means Clustering

The learned deck embeddings are divided into meta-deck clusters. This is a hybrid design: neural representation learning followed by classical unsupervised clustering.

### 4. CNN-style Spatial Features

The 4x7 board is processed with fixed 3x3 convolution-style kernels. These kernels capture spatial density and positioning patterns, but they are not trainable CNN layers.

For recording analysis, OpenCV additionally extracts:

- HSV saturation and brightness
- Canny edge density
- Visual occupancy per grid cell
- Frame-to-frame board activity

### 5. Feedforward Neural Network

The placement predictor combines:

- Autoencoder deck embedding
- CNN-style board features
- Game-flow or video-progress features

It outputs eight Softmax probabilities, one for each possible final placement.

```text
Top 4 probability = P(1st) + P(2nd) + P(3rd) + P(4th)
```

### 6. Training Techniques

- ReLU activation
- Softmax output
- Cross-entropy loss
- Mini-batch gradient descent
- Backpropagation
- L2 weight decay
- 80/20 hold-out evaluation

## Data and Evaluation

The current model is trained on 1,800 reproducible synthetic records:

- Training samples: 1,440
- Hold-out samples: 360
- Exact-placement accuracy: approximately 23.1%
- Top 4 accuracy: approximately 67.8%

The synthetic generator explicitly defines relationships between health, economy, board strength, actions, and placement. Its score distribution is centered to include both Top 4 and bottom 4 examples instead of producing an artificially high Top 4 rate. These hold-out results measure recovery of controlled synthetic rules and are not evidence of real TFT accuracy.

The recording analyzer is also a visual prototype. It normalizes board-activity signals against the 10th-to-90th percentile range of the selected recording and applies temperature calibration to reduce overconfident probabilities. Its action recommendation uses visual proxy state, not recognized HUD values. It is not trained on labeled TFT screenshots and does not identify exact champions, items, augments, health, or gold.

## Installation

Python 3.10 or later is recommended.

```powershell
git clone https://github.com/HISCHAN12/deeplearning.git
cd deeplearning
python -m pip install -r requirements.txt
```

Dependencies:

- NumPy
- OpenCV headless

## Run the Recording Analyzer

### Desktop App

```powershell
python -m src.tft_advisor.video_analysis_app
```

Then:

1. Select a TFT recording.
2. Choose the frame interval, normally five seconds.
3. Click `녹화 영상 분석`.
4. Wait for the whole video to be processed.
5. Open the generated report.

### Command Line

```powershell
python -m src.tft_advisor.video_analyzer "C:\Videos\tft_game.mp4" --interval 5
```

Generated files:

```text
outputs/video_analysis/.../video_analysis.json
outputs/video_analysis/.../video_analysis.html
```

See [`docs/video_analysis_ko.md`](docs/video_analysis_ko.md) for Korean instructions.

## Run the Other Prototypes

Train and print the synthetic-data demo:

```powershell
python -m src.tft_advisor.train_demo
```

Run the local browser advisor:

```powershell
python -m src.tft_advisor.live_advisor
```

Open `http://127.0.0.1:8000`.

Run the Windows manual-input overlay:

```powershell
python -m src.tft_advisor.overlay_advisor
```

## Tests

```powershell
python -m unittest discover -s tests
```

The tests cover:

- Placement and Top 4 probability outputs
- Game-state flow deltas
- State-aware recommendation adjustments
- Riot lockfile configuration
- Overlay display translation
- Video board-grid extraction
- End-to-end video analysis and report creation

## Repository Structure

```text
src/tft_advisor/
  game_state.py          Manual game state and flow deltas
  live_advisor.py        Local browser advisor
  models.py              NumPy autoencoder and feedforward model
  overlay_advisor.py     Windows manual-input overlay
  recommender.py         Training, clustering, prediction, recommendation
  state_encoder.py       Deck and board encoders
  synthetic_data.py      Reproducible synthetic training records
  train_demo.py          Console training demo
  video_analyzer.py      Offline OpenCV recording analysis
  video_analysis_app.py  Recording selection desktop UI
```

## Scope and Limitations

- Recording analysis is offline, not live.
- The complete video is sampled repeatedly, but the final report is shown after processing.
- The central screen area is assumed to contain the TFT board.
- Exact champions, items, augments, health, and gold are not automatically recognized.
- The visual analyzer uses handcrafted OpenCV features rather than a trained object detector.
- The board encoder uses fixed CNN-style filters, not a trainable CNN.
- The model is trained on synthetic data.
- Real performance claims require labeled mid-game TFT data and a separate real-data test set.
- Live automated game-state analysis and live recommendations are intentionally outside the implemented scope.

## Future Work

- Collect real match histories through the Riot Developer API.
- Build a labeled TFT screenshot and video-frame dataset.
- Train YOLO/CNN models for champion and item recognition.
- Add OCR for health, gold, level, and stage.
- Replace fixed kernels with a trainable CNN.
- Add an LSTM or Transformer for temporal recording analysis.
- Validate placement probabilities on real held-out games.

## Submission Checklist

- Make the repository public.
- Enable GitHub Pages.
- Replace `YOUR_VIDEO_ID`.
- Run the tests and recording analyzer.
- Record the approximately seven-minute presentation.
- Submit the public GitHub repository URL.

## References

- Goodfellow, Bengio, and Courville, *Deep Learning*.
- Chapters 6, 7, 8, 9, and 14.
- rndmagtanong/ph_tft.
- Mattbusel/tft-synapse.
- KennethLeeJE8/tftanalysis_set3.
