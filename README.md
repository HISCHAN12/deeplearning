# TFT Meta Deck Clustering and Action Advisor

## Team / Student

- Student ID: 22000708
- Name: Heechan Jung
- Course: Deep Learning Term Project

## Project Title

**A Deep Learning Framework for Teamfight Tactics Placement Prediction and Strategic Action Recommendation**

## Project Page and Video

- GitHub Pages URL: `https://hischan12.github.io/deeplearning/`
- YouTube demo video: `https://youtu.be/YOUR_VIDEO_ID`

Replace the YouTube link above after uploading the 7 minute demo video.

## Introduction

Teamfight Tactics (TFT) is a strategy game where players make sequential decisions about champions, traits, items, augments, economy, leveling, and board positioning. These decisions interact in highly non-linear ways, which makes TFT a strong real-world application for deep learning concepts.

This project proposes and prototypes a TFT advisor that clusters similar meta-decks, predicts final placement, and recommends the next strategic action. The system is designed around concepts from the deep learning lectures: deep feedforward networks, regularization, optimization, convolutional feature extraction, and autoencoders.

## Task and Method

The target task is:

1. Predict a player's final placement from an in-game board state.
2. Recommend the next action, such as leveling, rolling, holding economy, slamming an item, or repositioning the carry.

The implemented prototype follows this pipeline:

1. **Deck representation**
   Champion, trait, item, and augment information is encoded as a high-dimensional vector.

2. **Autoencoder embedding**
   A NumPy autoencoder compresses the high-dimensional deck vector into a low-dimensional latent embedding. Similar TFT compositions become closer in this embedding space.

3. **Meta-deck clustering**
   K-means clustering groups similar embeddings into meta-deck clusters.

4. **Spatial board feature extraction**
   A CNN-style board encoder applies fixed convolution kernels to the 4x7 TFT board grid to capture local positioning patterns. This is a deterministic feature extractor inspired by CNNs, not a trainable convolutional layer.

5. **Placement prediction**
   A feedforward neural network predicts the probability of each final placement from the deck embedding and board features. The Top 4 probability is the sum of the probabilities for placements 1 through 4.

6. **Action recommendation**
   Historical outcomes inside the nearest meta-deck cluster are analyzed. The advisor recommends the action with the best average placement and reports its improvement over the cluster baseline and its historical Top 4 rate.

## Experiments

The repository includes a reproducible synthetic TFT match-log generator. It is used because the Riot Developer API requires a personal API key and live data collection cannot be guaranteed during grading.

Run the prototype:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.tft_advisor.train_demo
```

Run the local live advisor / overlay demo:

```bash
python3 -m src.tft_advisor.live_advisor
```

Run the Windows always-on-top manual-input overlay:

```powershell
python -m src.tft_advisor.overlay_advisor
```

The overlay is a frameless, semi-transparent window that stays above TFT. It supports drag movement, compact mode, manual meta-deck and board-position selection, predicted placement, Top 4 probability, placement distribution, and action recommendation. Use TFT in windowed or borderless-windowed mode because exclusive fullscreen can cover desktop overlays.

Then open:

```text
http://127.0.0.1:8000
```

This web advisor refreshes every 2 seconds. If the Riot/League client lockfile is detected locally, it shows the client connection status. Windows common install paths and macOS paths are checked. The actual board-state recommendation is still driven by the selected sample state, which keeps the project demo reliable and avoids depending on fragile screen recognition.

The script generates match logs, uses an 80/20 train-test split, trains the autoencoder and placement predictor on the training split, clusters training decks, and prints hold-out evaluation metrics plus an example recommendation.

Run tests without extra test dependencies:

```bash
python3 -m unittest discover -s tests
```

## Result Analysis

The demo reports:

- Autoencoder reconstruction loss.
- Hold-out placement prediction accuracy.
- Hold-out Top 4 prediction accuracy.
- Model-predicted Top 4 probability.
- The nearest meta-deck cluster for a sample board.
- Recommended next action, expected placement improvement, and supporting cluster statistics.

With the current fixed seed, the demo produces approximately 0.26 exact-placement accuracy and 0.71 Top 4 accuracy on the 20% hold-out split. Because this is a course prototype, these numbers measure performance on generated data and must not be interpreted as performance on live TFT matches. With Riot API data, the same pipeline can be retrained and evaluated on real match histories.

The generated data intentionally contains archetype-specific action effects. The recommendation experiment therefore demonstrates whether the pipeline can recover known structure from controlled data; it does not establish that the recommended action causes better outcomes in real games.

## Implementation Choice

For the final demo, I recommend a **local web advisor / OBS browser overlay** rather than a direct game-client overlay.

Reasons:

- It is easier to implement and explain in a 7 minute course video.
- It avoids anti-cheat, screen scraping, and client integration risks.
- It still demonstrates the key idea: given a board state, predict placement and recommend an action.
- It can be shown beside the TFT client or captured as a transparent browser source.

The minimum viable demo should show:

1. A current board state.
2. Predicted placement distribution.
3. Nearest meta-deck cluster.
4. Recommended action.
5. Explanation based on cluster-level historical statistics.

## Submission Checklist

- Make the GitHub repository public.
- Replace all placeholder GitHub and YouTube links.
- Enable GitHub Pages for `index.html`.
- Run `python3 -m src.tft_advisor.train_demo`.
- Record a roughly 7 minute video using the outline in `docs/video_script_ko.md`.
- Submit the public GitHub repository link before the deadline.

## Running on Another Desktop

On the desktop where TFT is installed:

```bash
git clone https://github.com/HISCHAN12/deeplearning.git
cd deeplearning
python3 -m pip install -r requirements.txt
python3 -m src.tft_advisor.live_advisor
```

Open `http://127.0.0.1:8000` in a browser. If Riot/League/TFT is running and the local lockfile can be found, the page displays the client detection status. The recommendation itself uses the selected sample state so the project demo remains reliable.

Expected terminal output:

```text
Starting TFT live advisor...
Training local prototype model. This usually takes a few seconds.
Model ready.
TFT live advisor running at http://127.0.0.1:8000
Press Ctrl+C to stop.
```

If port 8000 is already in use:

```bash
TFT_ADVISOR_PORT=8001 python3 -m src.tft_advisor.live_advisor
```

On Windows PowerShell, use:

```powershell
$env:TFT_ADVISOR_PORT=8001
python -m src.tft_advisor.live_advisor
```

If League of Legends is installed in a custom folder, specify its lockfile:

```powershell
$env:TFT_RIOT_LOCKFILE="D:\Riot Games\League of Legends\lockfile"
python -m src.tft_advisor.live_advisor
```

See `docs/other_desktop_setup_ko.md` for the Korean step-by-step guide.

## References

- Goodfellow, Bengio, and Courville, *Deep Learning*.
- Chapter 6: Deep Feedforward Networks.
- Chapter 7: Regularization for Deep Learning.
- Chapter 8: Optimization for Training Deep Models.
- Chapter 9: Convolutional Networks.
- Chapter 14: Autoencoders.
- rndmagtanong/ph_tft: TFT match data collection and placement prediction.
- Mattbusel/tft-synapse: AI-powered TFT advisor concept.
- KennethLeeJE8/tftanalysis_set3: Similarity and win-rate based TFT recommendation.

## Scope and Limitations

- This is a manual-input deep learning advisor prototype with both a local web UI and a Windows always-on-top overlay.
- The current board, champions, items, and augments are selected manually in the web UI or overlay.
- Riot client detection reports client availability only; it does not extract the live TFT board.
- The board encoder uses fixed CNN-style filters. The autoencoder and feedforward placement predictor are the trainable neural components.
- The current experiment uses reproducible synthetic data. Real-match claims require Riot API data and a separate real-data test set.
