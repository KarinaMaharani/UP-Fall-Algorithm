# UP-Fall-Algorithm

A re-implementation of the **UP-Fall Detection benchmark** that runs on any fall dataset. The
same code runs on the original UP-Fall, on the **UR Fall Detection Dataset**, and on **FAMS**
(the RA-Pak-Edy recordings: 3 phone IMUs and 4 webcams per session). Each dataset gets one small
adapter; the rest of the pipeline is shared.

It reproduces two published benchmarks:

| Track | Paper | What it benchmarks |
|---|---|---|
| **ARC** (`run`, `cnn`) | Martínez-Villaseñor *et al.*, "UP-Fall Detection Dataset: A Multimodal Approach", *Sensors* 19(9):1988, 2019 | Windowing (1/2/3 s), 18 features, 5-method feature selection, then RF / SVM / MLP / kNN on modality combinations. Also a CNN on raw frames. |
| **Skeleton** (`skeleton`) | Koffi *et al.*, "An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection", 2024 (repo `3D_skeletons-UP-Fall-Dataset`) | Impact detection from skeleton joints: SVM / MLP / RF / KNN / NB / SGD on frames, and LSTM / BiLSTM / CNN / ST-GCN on sequences. |

> **About the original repository.** The `3D_skeletons-UP-Fall-Dataset` zip contains only a
> README and `SUBJECT1-5.zip` with 82 skeleton CSVs. The `src/`, `examples/` and `notebooks/`
> that its README describes are not in the release, so everything here was written from the two
> papers. Where a paper leaves a detail open, the choice made here is listed in
> [§13 Deviations from the papers](#13-deviations-from-the-papers).

---

## Contents

**Getting it running**
1. [The datasets at a glance](#1-the-datasets-at-a-glance)
2. [Installation](#2-installation)
3. [Installing the datasets](#3-installing-the-datasets)
4. [Running the benchmarks](#4-running-the-benchmarks)
5. [Checking progress, stopping and resuming](#5-checking-progress-stopping-and-resuming)
6. [Results](#6-results)

**Reference**

7. [How the pipeline works](#7-how-the-pipeline-works)
8. [Dataset details](#8-dataset-details)
9. [Commands](#9-commands)
10. [Configuration](#10-configuration)
11. [Adding a new dataset](#11-adding-a-new-dataset)
12. [Benchmark details](#12-benchmark-details)
13. [Deviations from the papers](#13-deviations-from-the-papers)
14. [Methodological notes for a thesis](#14-methodological-notes-for-a-thesis)
15. [Known issues and troubleshooting](#15-known-issues-and-troubleshooting)
16. [Project structure](#16-project-structure)
17. [References](#17-references)

---

## 1. The datasets at a glance

| dataset | where it comes from | installed by | benchmarks |
|---|---|---|---|
| **UR Fall** | public website | the pipeline downloads it (~110 MB) | ML benchmark (`run`), optional CNN and skeleton |
| **FAMS** (19 sessions, one person each) | the `dataset` account on the server | you copy the zips, the pipeline unpacks them | ML benchmark (`run`), optional CNN and skeleton |
| **UP-Fall, 3D skeletons** (Koffi et al. repo) | `SUBJECT*.zip` from the repo zip | you copy the zips | skeleton benchmark (`skeleton`) |
| **UP-Fall, original** (Martínez-Villaseñor et al.) | HAR-UP website | you download it | ML benchmark (`run`) |

Every dataset lives in `data/<name>/`. Derived data (optical flow, poses, window tables) goes to
`cache/<name>/`, and results to `results/<name>/`. All three folders are git-ignored.

In the commands below, `[laptop]` means WSL on your laptop and `[server]` means after
`ssh -p 51022 karina@152.118.31.34`. For another server, change the user, host and port.

---

## 2. Installation

### 2.1 What the server needs

- Linux with an NVIDIA GPU. The GPU is only used for pose extraction and the optional CNN and
  skeleton models; everything else runs on the CPU.
- About **60 GB free** in your home folder: FAMS is 16 GB of zips plus about 16 GB unpacked,
  and the cache grows to a few GB. Check with `df -h ~`.
- Internet access, for Python packages and the UR Fall download.

### 2.2 Copy the code

```bash
# [laptop] the cd && guard means nothing is sent if the folder is wrong
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" --exclude data/ --exclude cache/ --exclude results/ --exclude __pycache__/ ./ karina@152.118.31.34:~/upfall/
```

Re-run this whenever the code changes; it never touches `data/`, `cache/` or `results/` on the
server. Never run rsync with `./` from another folder: it sends whatever folder you're in. Don't
re-send while the pipeline is running, because bash reads scripts as it goes and a script changed
mid-run can break the run. Git Bash on Windows has no `rsync`, so use WSL.

### 2.3 Python environment (no admin rights needed)

Use Python 3.12, the version the server runs on. If `python3.12` isn't installed, `uv` downloads
it into your home folder:

```bash
# [server]
cd ~/upfall
pip install uv
uv python install 3.12
uv venv --python 3.12 venv312
source venv312/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

Check it, which should print `3.12.x ... True`:

```bash
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
```

The scripts activate `venv312` by themselves (or `venv` if there's no `venv312`).
`ffmpeg` comes with the `imageio-ffmpeg` package, so no system install is needed.

### 2.4 On a laptop instead

```bash
pip install -r requirements.txt
python -m upfall_bench run -d configs/datasets/synthetic.yaml -b configs/benchmarks/quick.yaml   # ~1 min check
```

By default every command stays light: 2 CPU cores, 2 torch threads and a GPU batch of 4
(see `resources:` in [§10](#10-configuration)). Add `-s resources.device=cpu` to keep the GPU free
entirely. Run one command at a time on a laptop; full benchmarks belong on the server.

---

## 3. Installing the datasets

### 3.1 UR Fall: nothing to do

The pipeline downloads it in step 4.1, skipping files that already exist. By hand:

```bash
# [server]
cd ~/upfall && python -m upfall_bench download -d configs/datasets/urfall.yaml
```

### 3.2 FAMS: copy the zips from the `dataset` account

The originals are in `/home/dataset/FAMS` (`SF01.zip` … `SF19.zip`, some already unpacked as
`SF16`, `SF17` and `SF18`). Copy them into `data/fams`:

```bash
# [server] if your karina account can read the dataset folder:
mkdir -p ~/upfall/data && cp -r /home/dataset/FAMS ~/upfall/data/fams

# otherwise pull it over ssh (asks for the dataset password):
rsync -avP -e "ssh -p 51022" dataset@152.118.31.34:~/FAMS/ ~/upfall/data/fams/
```

**Don't unzip them yourself into one folder.** Every export zip has `data/`, `videos/` and
`cameras.json` at its top level, so unzipping several into the same folder merges them into one
session. The pipeline unpacks each zip into its own folder (`SF01.zip` → `SF01/`) with Python, so
`unzip` isn't needed. To unpack by hand anyway:

```bash
# [server]
cd ~/upfall && python - <<'EOF'
import glob, os, zipfile
for z in sorted(glob.glob("data/fams/*.zip")):
    dest = z[:-len(".zip")]
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            target = os.path.join(dest, info.filename)
            if not (os.path.exists(target) and os.path.getsize(target) == info.file_size):
                zf.extract(info, dest)
    print("extracted", z, "->", dest)
EOF
```

Check that all 19 sessions load, one line per person:

```bash
# [server]
cd ~/upfall && python -m upfall_bench inspect -d configs/datasets/fams.yaml 2>&1 | grep "labelled segments"
```

### 3.3 UP-Fall, 3D-skeleton release

Copy the five `SUBJECT*.zip` files from the `3D_skeletons-UP-Fall-Dataset` repo zip. They are
read as they are, with no need to unpack:

```bash
# [laptop] from the folder where you unpacked the repo zip
rsync -avP -e "ssh -p 51022" SUBJECT*.zip karina@152.118.31.34:~/upfall/data/upfall_skeleton/
```

### 3.4 UP-Fall, original consolidated dataset

Download the *Consolidated dataset* from <https://sites.google.com/up.edu.mx/har-up/> into
`~/upfall/data/upfall/`. The adapter expects the per-trial CSV files, in any folder layout:

```
data/upfall/Subject1/Activity1/Trial1/Subject1Activity1Trial1.csv
                                     /Subject1Activity1Trial1Camera1.zip   (optional, for CAM)
                                     /Subject1Activity1Trial1Camera2.zip   (optional)
```

The camera zips are large and only needed for the CAM experiments; IR, IMU and EEG work without
them. **This adapter has not been run on the real files yet.** Run `inspect` first; it should
report 17 subjects, 11 activities and around 560 recordings. If the site gives you one big
`CompleteDataSet.csv` instead of per-trial files, the adapter needs a change.

```bash
# [server]
cd ~/upfall && python -m upfall_bench inspect -d configs/datasets/upfall.yaml
```

### 3.5 Single RA-Pak-Edy sessions (optional)

`configs/datasets/ra_pak_edy.yaml` reads loose session exports from `data/ra_pak_edy/`, one
folder per unzipped export. It was used for the first test runs on the `francis` session; FAMS
uses the same settings for all 19 sessions.

---

## 4. Running the benchmarks

### 4.1 UR Fall and FAMS: one command

```bash
# [server]
cd ~/upfall && setsid nohup bash scripts/run_server.sh >> run_all.log 2>&1 < /dev/null &
echo "started, pid $!"
```

It runs in the background, keeps going after you log out, and does, in order:

1. download UR Fall; unpack the FAMS zips, each into its own folder
2. `inspect` both datasets
3. optical flow from the videos (CAM experiments)
4. poses from the videos (SKELETON experiments, GPU): the slow step, overnight for FAMS
5. the ML benchmark for UR Fall, then for FAMS

Everything is resumable: if it stops, run the same command again and every finished step is
skipped. Run only one copy at a time. It uses all CPU cores by default; for fewer, start it as
`N_JOBS=8 setsid nohup bash scripts/run_server.sh ...`.

### 4.2 UP-Fall

After `inspect` works (§3.3 and §3.4):

```bash
# [server] 3D-skeleton benchmark (ML + LSTM/BiLSTM/CNN/ST-GCN), takes minutes
cd ~/upfall && python -m upfall_bench skeleton -d configs/datasets/upfall_skeleton.yaml -b configs/benchmarks/koffi2024_skeleton.yaml

# [server] original UP-Fall, ML benchmark (paper protocol), resumable
cd ~/upfall && setsid nohup bash scripts/run_resumable.sh configs/datasets/upfall.yaml results/upfall/upfall2019 -s resources.n_jobs=$(nproc) >> upfall_bench.log 2>&1 < /dev/null &
```

For the CAM experiments of the original UP-Fall, run
`python -m upfall_bench extract-flow -d configs/datasets/upfall.yaml` first (needs the camera zips).

### 4.3 Optional extra benchmarks

```bash
# [server] vision CNN on raw frames (Sec. 5.6 of the paper)
python -m upfall_bench cnn -d configs/datasets/urfall.yaml
python -m upfall_bench cnn -d configs/datasets/fams.yaml

# [server] skeleton benchmark on poses extracted from your videos (after §4.1)
python -m upfall_bench skeleton -d configs/datasets/fams.yaml -b configs/benchmarks/koffi2024_skeleton.yaml -s skeleton.label=fall
python -m upfall_bench skeleton -d configs/datasets/urfall.yaml -b configs/benchmarks/koffi2024_skeleton.yaml -s skeleton.label=fall
```

On small datasets the CNN with the paper's 5 epochs only learns the majority class (UR Fall:
F1 0.16); with `-s cnn.epochs=30 -s cnn.lr=0.01` it reached F1 0.84. Report which setting you
used. Use `skeleton.label=fall` for UR Fall: `impact` looks for the acceleration peak inside the
fall class, and since that class is *lying after the fall*, the impact usually happens just
before it and is missed.

### 4.4 Other protocols and settings

```bash
# subject-independent: no window of a test person in training, selection on training data only
bash scripts/run_resumable.sh configs/datasets/fams.yaml results/fams/loso -b configs/benchmarks/subject_independent.yaml -s evaluation.protocol=loso
# quick sanity run: one window size, 2 rounds
python -m upfall_bench run -d configs/datasets/urfall.yaml -b configs/benchmarks/quick.yaml --out results/urfall/quick
```

Use a new results folder for each different setting (see [§10](#10-configuration)).

---

## 5. Checking progress, stopping and resuming

```bash
# [server] read-only
cd ~/upfall
echo "=== running? ==="; ps aux | grep -E "run_server|run_resumable|upfall_bench" | grep -v grep | cut -c1-120 || echo "NOT RUNNING"
echo "=== steps finished ==="; grep -aE "flow_fams: done|pose_urfall: done|pose_fams: done|no progress in 20|FINISHED|all done" run_all.log | tail -6
echo "=== FAMS poses: $(ls cache/fams/derived/pose 2>/dev/null | wc -l) videos done, $(ls cache/fams/derived/pose_chunks 2>/dev/null | wc -l) in progress ==="
echo "=== crash retries so far: $(grep -ac 'ended with exit code' logs/pose_fams.log 2>/dev/null) ==="
echo "=== FAMS benchmark parts: $(ls results/fams/upfall2019/parts 2>/dev/null | wc -l) ==="
echo "=== disk ==="; df -h ~ | tail -1
echo "=== last lines ==="; tail -c 400 run_all.log | tr '\r' '\n' | tail -3 | cut -c1-150
```

- `ended with exit code 139, retrying` is normal (see [§15](#15-known-issues-and-troubleshooting)).
  `no progress in 20 tries` is not.
- The pipeline is finished when the log ends with `all done`.
- Per-step logs are in `~/upfall/logs/`; each benchmark folder also has `resumable.log`.

**Stop everything:**

```bash
# [server]
pkill -f run_server.sh; pkill -f run_resumable.sh; pkill -f upfall_bench; sleep 3
ps aux | grep upfall_bench | grep -v grep || echo "stopped"
```

**Resume:** run the start command from §4.1 again. Saved work is kept: finished videos, the
pose chunks of the video in progress, and finished benchmark experiments.

---

## 6. Results

| benchmark | folder on the server |
|---|---|
| UR Fall ML | `~/upfall/results/urfall/upfall2019/` |
| FAMS ML | `~/upfall/results/fams/upfall2019/` |
| UP-Fall original ML | `~/upfall/results/upfall/upfall2019/` |
| skeleton and CNN runs | `~/upfall/results/<dataset>/skeleton_<date>/`, `cnn_<date>/` |

Bring them to your laptop:

```bash
# [laptop]
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" karina@152.118.31.34:~/upfall/results/ ./results/
```

An ML benchmark folder contains:

| file | content |
|---|---|
| `results.md` | best configuration per task, Tables 13 and 14, all figures |
| `table13_best_model_per_window.csv` | per modality × window: best model by mean F1 (paper Table 13) |
| `table14_best_window_per_model.csv` | per modality × model: best window by mean F1 (paper Table 14) |
| `summary.csv` | mean and std of every metric for every task × experiment × window × model |
| `raw_metrics.csv` | one row per round, for your own statistics (e.g. Wilcoxon tests) |
| `parts/` | one CSV per finished experiment; what resuming skips |
| `selection/*.json` | selected features, full voted ranking, votes, incremental curve |
| `confusion/*.npz` | confusion matrix summed over the test rounds, per combination |
| `figures/selection_*.png` | incremental accuracy curves (paper Figs. 8–14) |
| `figures/f1_*.png` | F1 per modality and model |
| `figures/confusion_*.png` | confusion matrix of the best configuration (paper Fig. 16) |
| `config.yaml`, `log.txt`, `resumable.log` | the settings used and the console logs |

`cnn` writes `cnn_results.json` and `cnn_confusion.npz`; `skeleton` writes `results.md`,
`skeleton_table.csv` and `raw_metrics.csv`. `inspect` writes `results/<dataset>/inspect_*.csv`.
To rebuild the tables and figures of a finished run: `python -m upfall_bench report --run <folder>`.

---

## 7. How the pipeline works

The UP-Fall paper follows the *activity recognition chain* (ARC, Fig. 7):

```
 sensor/video files
        │  dataset adapter (one per dataset)          upfall_bench/datasets/*.py
        ▼
 Recording(subject, activity, trial, streams{IMU, CAM, ...}, label timeline, videos)
        │  extract-flow / extract-pose                upfall_bench/derive/
        ▼        (adds CAM / SKELETON streams from the videos, cached)
 Windowing: 1/2/3 s, 50 % overlap, sample-and-hold to 18 Hz        arc/windowing.py
        ▼
 Feature extraction: 12 temporal + 6 frequency per channel           arc/features.py
                     (+ mean 20×20 optical-flow vector per camera)
        ▼
 Feature selection: 5 rankers → top-100 each → vote → RF curve       arc/selection.py
        ▼
 Models: RF, SVM, MLP, kNN (Table 12)                                arc/models.py
        ▼
 Evaluation: 10 × random 70/30, acc / prec / sens / spec / F1        arc/evaluation.py
        ▼
 Report: Table 13, Table 14, Fig. 8-14 curves, confusion matrix      arc/report.py
```

These steps run for **every window size × every experiment (modality combination) × every
task**: `multiclass` (all activity labels, as in the paper) and `binary` (fall vs. everything else).

**Data model.** Every adapter produces `Recording` objects
([upfall_bench/core.py](upfall_bench/core.py)):

| field | meaning |
|---|---|
| `rid, subject, activity, trial` | one UP-Fall *SubjectXActivityYTrialZ* |
| `streams: {name: Stream}` | each sensor: `t_ms` (N), `values` (N × C), `channels`, `modality` tag (`IMU`, `CAM`, `IR`, `EEG`, `DEPTH`, `SKELETON`, …), `kind` (`signal` → 18 features, `frame_vector` → mean) |
| `label_t_ms, labels` | label timeline: one label per sample, or only the change points (sample-and-hold) |
| `videos: {cam: VideoSource}` | video file or image zip, and how frame index maps to time |

Every stream of a recording uses the same millisecond clock. That is all the pipeline needs to
know about a dataset.

---

## 8. Dataset details

### 8.1 UR Fall (`urfall`)

<https://fenix.ur.edu.pl/mkepski/ds/uf.html>. 30 falls and 40 ADL sequences by 5 people, with no
subject IDs per sequence. The streams are:

| stream | modality | channels |
|---|---|---|
| `acc` | `IMU` | ax, ay, az, sv_total (waist accelerometer, ~60 Hz, no gyroscope) |
| `depth_feat` | `DEPTH` | HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40 |
| `cam0_flow` (after `extract-flow`) | `CAM` | 400 optical-flow cells |
| `cam0_pose` (after `extract-pose`) | `SKELETON` | COCO-17 x, y |

`download` fetches `*-acc.csv`, `*-data.csv` (frame times) and `*-cam0.mp4` for every sequence,
plus `urfall-cam0-falls.csv` and `urfall-cam0-adls.csv`. The mp4 shows depth and RGB side by side
(640 × 240), and the adapter keeps the RGB half.

**Classes.** UR Fall labels every frame with a posture: −1 not lying, 0 falling, 1 lying. As the
dataset page says, the `0` frames (a temporary pose while falling) are not used in
classification, so windows labelled `0` are dropped (`skip_labels: [0]`). The rest:

| id | class | source |
|---|---|---|
| 1 | upright | −1 anywhere |
| 3 | **lying after fall** | 1 in a fall sequence: the fall (positive) class |
| 5 | lying (ADL) | 1 in an ADL sequence: not a fall |

Experiments: `IMU`, `DEPTH`, `CAM`, `IMU+DEPTH`, `IMU+CAM`, `DEPTH+CAM`, `IMU+DEPTH+CAM`,
`SKELETON`, `IMU+SKELETON`.

### 8.2 FAMS and RA-Pak-Edy sessions (`imu_session`)

Each session is one export of the `data-collector-IMU-mobile` app:

```
cameras.json                       started_at_ms of every webcam (video clock)
videos/<sid>_camN_video_sync.webm  4 × Logitech C270, VP9, 1280×720
data/manifest.json                 subject, operator, label_timeline (label change points)
data/<sid>_{waist,chest,thigh_right}_consolidated.csv
     timestamp_ms, acc_x_g, acc_y_g, acc_z_g, gyro_x_degs, gyro_y_degs, gyro_z_degs, label_id, sample_kind, …
```

The adapter:

* finds every session folder under `root`, nested or not;
* uses `timestamp_ms` as the time axis (the exports are irregular, ~96 Hz true) and drops the
  `sample_kind != 0` rows, which are held or placeholder samples;
* cuts each session at the label change points in `manifest.json → integrity.label_timeline`.
  Each run of a non-zero label becomes one `Recording`; label `0` (idle) is dropped;
* takes the subject from the manifest, or the folder name when the manifest's is blank;
* adds `cam1`–`cam4` with frame time = `started_at_ms + i / video_fps`. `cam4` is rotated, since
  it was filmed sideways. A camera without a start time in `cameras.json` is skipped with a warning.

| stream | modality | channels |
|---|---|---|
| `waist`, `chest`, `thigh_right` | `IMU` | acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (°/s) |
| `camN_flow` (after `extract-flow`) | `CAM` | 400 optical-flow cells |
| `camN_pose` (after `extract-pose`) | `SKELETON` | COCO-17 x, y |

**Labels.** [configs/datasets/ra_pak_edy.yaml](configs/datasets/ra_pak_edy.yaml) holds the
protocol from the last table of the informed-consent form: 38 activity names, of which 22 are
FALL (`fall_classes: [2, 4, 6, 8, 10, 14, 16–19, 21–32]`) and 16 are ADL. Each FALL segment
covers the whole labelled stretch, so standing before and lying after the fall count as fall too.

**FAMS** ([configs/datasets/fams.yaml](configs/datasets/fams.yaml)) reuses all of that through
`extends: ra_pak_edy.yaml`, with `root: data/fams`, all four cameras, and poses at 5 frames per
second. It has 19 sessions, one person each.

Experiments: `IMU` (all three phones), `waist`, `chest`, `thigh`, `waist+chest`, `CAM`, `cam1`,
`IMU+CAM`, `SKELETON`, `IMU+SKELETON`.

### 8.3 Original UP-Fall (`upfall`)

Reads `SubjectXActivityYTrialZ.csv`: timestamp, the 42 signals in Table 6 order, then Subject,
Activity, Trial and Tag. The label is the last column, which inside a fall trial switches between
standing, falling and lying. Camera zips are read directly. The experiments are the paper's
seven: `IR`, `IMU`, `IMU+EEG`, `IR+IMU+EEG`, `CAM`, `IR+CAM`, `IMU+EEG+CAM`. Running it lets you
compare against Tables 13–14 of the paper before trusting the numbers on a new dataset.

### 8.4 Improved 3D-skeleton UP-Fall (`upfall_skeleton`)

82 files `C{cam}S{subj}A{act}T{trial}.csv`, each 100 frames × 33 MediaPipe joints × (X, Y, Z)
plus an impact label (1/0), for fall activities 1–5 only. The adapter reads the zips directly and
tolerates the release's misnamed files (`C2S1_A1_T2`, `CS3A4T2`, `3S3A4T1`) and inconsistent
headers (`LLABEL`, missing header).

### 8.5 Synthetic (`synthetic`)

A fake dataset with the UP-Fall structure, for tests and for trying options quickly.

---

## 9. Commands

```
python -m upfall_bench <command> -d <dataset.yaml> [-b <benchmark.yaml>] [-s key=value ...]
```

| command | does |
|---|---|
| `list` | registered adapters and the available dataset configs |
| `download` | download the dataset (UR Fall only) |
| `inspect` | recordings, subjects, streams and rates; per-activity duration and acceleration peak; time per label class |
| `extract-flow` | Horn–Schunck optical flow → 20×20 magnitude per frame → `CAM` stream (cached) |
| `extract-pose` | Keypoint R-CNN (or MediaPipe) skeletons → `SKELETON` stream (cached, saved in chunks) |
| `windows` | only build and cache the window feature tables, and print class counts |
| `run` | **the UP-Fall ML benchmark**: selection + RF/SVM/MLP/kNN for every window × experiment × task |
| `cnn` | the vision CNN of Sec. 5.6 on 28×28 frames |
| `skeleton` | the 3D-skeleton benchmark (ML on frames + LSTM/BiLSTM/CNN/ST-GCN on sequences) |
| `report --run <folder>` | rebuild the tables and figures of a finished run |

`run` writes to `results/<dataset>/upfall2019/` unless you give `--out`; `run --rebuild`
recomputes the cached windows; `extract-* --overwrite` recomputes cached videos.

The two scripts wrap these commands: `scripts/run_server.sh` (the whole UR Fall + FAMS pipeline)
and `scripts/run_resumable.sh <dataset.yaml> <out_dir> [args]` (one `run`, restarted after a
crash, stopping if the settings don't match the folder).

---

## 10. Configuration

The config is built from layers, in order:

1. [configs/benchmarks/defaults.yaml](configs/benchmarks/defaults.yaml), which documents every option;
2. a **benchmark** file (`-b`, default `upfall2019.yaml`);
3. the base of the dataset file, if it says `extends: other.yaml`;
4. the **dataset** file (`-d`);

then any number of `-s dotted.key=value` overrides (values are parsed as YAML).

| benchmark file | purpose |
|---|---|
| `upfall2019.yaml` | faithful paper protocol (default) |
| `quick.yaml` | one window, 2 rounds, short selection curve: minutes instead of hours |
| `subject_independent.yaml` | group split by trial (or `loso`), selection inside each training split, standardised features |
| `koffi2024_skeleton.yaml` | 3D-skeleton benchmark settings |

Examples:

```bash
# only 2-s windows, binary task, 5 rounds
python -m upfall_bench run -d configs/datasets/urfall.yaml -s arc.windows=[2] -s arc.tasks=[binary] -s evaluation.rounds=5 --out results/urfall/test
# keep the IMUs at their own rate instead of the paper's 18 Hz consolidation
python -m upfall_bench run -d configs/datasets/fams.yaml -s arc.resample_hz=native --out results/fams/native
```

**A results folder resumes only with the settings it was started with.** The experiment list and
the `resources:` block may change between restarts; anything else stops the run with
`started with a different configuration`. Use a new `--out` folder for each different setting.

**Resources.** The `resources:` block keeps a run from taking over the machine:

| key | default | meaning |
|---|---|---|
| `n_jobs` | 2 | CPU cores for Random Forest, feature selection and numpy/BLAS |
| `torch_threads` | 2 | CPU threads for PyTorch |
| `device` | auto | `auto` = GPU if present, `cpu` = never touch the GPU |
| `gpu_batch` | 4 | frames per pose-model call |
| `gpu_memory_fraction` | null | share of GPU memory PyTorch may use (`null` = no cap) |

`run_server.sh` sets `n_jobs` to all cores of the server.

**Subsets.** `-s "dataset.include=['fall-0[1-4]', 'adl-0[1-4]']"` keeps only recordings whose id
matches a glob pattern, handy for trying `extract-pose` or `cnn` on a few videos first. Quote the
patterns, because `[` is YAML syntax.

**Experiments.** The `experiments:` section of a dataset YAML maps a name to a list of tokens. A
token matches a **modality tag** (`IMU`), a **stream name** (`waist`) or a **glob** on stream
names (`cam*_flow`). An experiment whose streams don't exist (e.g. CAM before `extract-flow`) is
skipped with a message.

```yaml
experiments:
  IMU: [IMU]
  waist-only: [waist]
  front-cams: ["cam1_flow", "cam2_flow"]
  IMU+CAM: [IMU, CAM]
```

---

## 11. Adding a new dataset

This is the only place where dataset-specific code lives.

1. **Copy the template**: [upfall_bench/datasets/_template.py](upfall_bench/datasets/_template.py)
   → `upfall_bench/datasets/my_dataset.py`. Rename the class and uncomment `@register("my_dataset")`.
2. **Implement `load_recordings()`**. It returns one `Recording` per trial: fill in `subject`,
   `activity`, `trial`, the label timeline, one `Stream` per sensor with a modality tag, and
   optionally `VideoSource`s. Keep timestamps in ms on one clock per recording.
3. **Register it**: add `from . import my_dataset` to
   [upfall_bench/datasets/__init__.py](upfall_bench/datasets/__init__.py), or skip steps 1–3 and
   point the YAML at your file: `adapter: "path/to/my_dataset.py:MyDatasetAdapter"`.
4. **Write the config**: copy [configs/datasets/_template.yaml](configs/datasets/_template.yaml)
   and set `root`, `fall_classes`, `classes` and `experiments`. Every key under `dataset:` is
   available in the adapter as `self.dcfg[...]`. A dataset recorded with the same protocol as an
   existing one can just say `extends: that_dataset.yaml` and override `name` and `root`, as
   `fams.yaml` does.
5. **Check it**: run `inspect` and confirm the durations, rates and time per class. Then run
   `windows`, then `run -b configs/benchmarks/quick.yaml`, then the full `run`.

Optional overrides: `default_classes()`, `default_fall_classes()`, `download()`. Labels listed in
`dataset.skip_labels` are left out of every window, as UR Fall does with its transition frames.

Optical flow, pose extraction, the CNN and the skeleton benchmark work for any adapter that fills
`videos`. A camera given as `VideoSource(cam, path, t0_ms=…, fps=…)` is decoded at that constant
rate with ffmpeg; one given with `frame_times_ms=[…]` uses the exact frame times. `crop` and
`rotate` are applied first.

Tips:

* **Session-style data** (one long take, labels switched live) → cut it into one Recording per
  label run, as [imu_session.py](upfall_bench/datasets/imu_session.py) does.
* **Trial-style data** (one file per trial) → one Recording per file, as
  [upfall.py](upfall_bench/datasets/upfall.py) does.
* **Precomputed per-frame features** shipped with a dataset → add them as a `signal` stream with
  their own modality tag, like UR Fall's `DEPTH`.

---

## 12. Benchmark details

### 12.1 Windowing (Sec. 5.2)

* Windows of 1, 2 and 3 s with 50 % overlap, over each recording's labelled span.
* Each stream is first put on a uniform grid by sample-and-hold. The paper put every device on
  the camera clock at ~18.4 Hz with "drop-sampling interpolation" (repeat the last value), which is
  `resample_hz: 18`; `native` keeps each stream at its own rate.
* A window's label is the **most frequent label** inside it, as in the paper. The binary label
  is 1 if that label is in `fall_classes`. Windows whose label is in `skip_labels` are dropped.
* A stream with samples over less than 50 % of a window gives NaN features; those windows are
  dropped per experiment.

### 12.2 Features (Tables 10 and 11)

Computed for every channel of every `signal` stream.

| feature | definition used |
|---|---|
| mean, std, median, max, min | usual |
| rms | √(mean x²) |
| zero_crossings | number of sign changes of the raw signal |
| skewness, kurtosis | scipy (Fisher kurtosis, biased) |
| q1, q3 | 25th and 75th percentiles |
| autocorrelation | lag-1 autocorrelation of the mean-removed signal |
| mean_freq | Σ f·P(f) / Σ P(f), with P = \|FFT\|² |
| median_freq | frequency that splits the spectral power in half |
| entropy | Shannon entropy (bits) of the normalised power spectrum |
| energy | Σ \|X_k\|² / N |
| principal_freq | frequency of the largest non-DC power peak |
| spectral_centroid | Σ f·\|X(f)\| / Σ \|X(f)\| |

**Vision (Sec. 5.2, Eq. 1).** For every pair of consecutive frames: Horn–Schunck flow (u, v),
then d = √(u² + v²), resized to 20 × 20 and flattened to 400 values. These vectors are
**averaged** inside the window, giving 400 features per camera.

### 12.3 Feature selection (Sec. 5.3)

The paper's 8 steps, with Python versions of its Weka evaluators:

1. Drop features with missing values or zero variance.
2. Score the features with five methods: `cfs` (greedy forward CFS merit, ≈ CfsSubsetEval),
   `correlation` (≈ CorrelationAttributeEval), `relieff` (≈ ReliefFAttributeEval, k = 10),
   `oneattr_tree` (CV accuracy of a depth-3 tree on one feature, ≈ ClassifierAttributeEval +
   DecisionTable) and `mutual_info`.
3. Take the top 100 of each method.
4. Count how many of these subsets each feature appears in.
5. Keep the features with ≥ 2 votes.
6. Sort them by votes.
7. Incremental analysis: 3-fold CV accuracy of RF (10 trees) on the first k features, k = 1 … 80.
8. Keep the smallest k within 0.5 accuracy points of the best k (the paper picked k by eye).

### 12.4 Models (Table 12)

Table 12's parameter names are scikit-learn's, so they map one to one:

| model | scikit-learn |
|---|---|
| RF | `RandomForestClassifier(n_estimators=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True)` |
| SVM | `SVC(C=1.0, kernel="rbf", gamma="auto" (= 1/features), shrinking=True, tol=1e-3)` |
| MLP | `MLPClassifier((100,), relu, solver="adam", alpha=1e-4, batch_size=min(200, n), lr=1e-3, shuffle, tol=1e-4, β=(0.9, 0.999), ε=1e-8, max_iter=10)` |
| kNN | `KNeighborsClassifier(n_neighbors=5, leaf_size=30, metric="euclidean")` |
| NB, SGD | `GaussianNB()`, `SGDClassifier()` (the skeleton benchmark's extra models) |

The table lists "solver = stochastic gradient" with Adam's β₁, β₂ and ε, so this is sklearn's
`adam` solver. No feature scaling is mentioned, so the default is `scaler: none`.

### 12.5 Evaluation (Sec. 5.4)

* 10 rounds of a random, stratified 70 / 30 split of the windows; the reported value is mean ± std.
* Metrics: accuracy, precision, sensitivity, specificity and F1 (Eq. 2–6). For more than two
  classes, precision, sensitivity, specificity and F1 are macro-averaged one-vs-rest, which is why
  the paper reports ~99.5 % specificity next to ~70 % sensitivity. For `binary`, the positive
  class is *fall*.
* Table 13: best model per modality × window. Table 14: best window per modality × model. Both
  pick by mean F1.
* Other protocols: `kfold`, `group_holdout` (split by trial) and `loso` (leave one subject out).

### 12.6 Vision CNN (Sec. 5.6, Fig. 17)

Input: a 28 × 28 × 3 frame, zero-centred. Three blocks of conv 3×3 → ReLU → max-pool 2×2 with 8,
16 and 32 filters, then a fully connected layer and softmax. SGD with momentum 0.9 (MATLAB
`sgdm`), learning rate 0.001, L2 0.004, 5 epochs, mini-batch 100. 2/3 train and 1/3 test; within
the training part, 5-fold CV picks the fold model with the best validation accuracy, which is then
scored on the test set. One camera, as in the paper (`cnn.cameras`). Frames are labelled with the
label at their timestamp.

### 12.7 3D-skeleton benchmark (Koffi et al. 2024)

* **ML** classifies every frame from all joint coordinates (99 values for MediaPipe-33) with
  SVM, MLP, RF, KNN, NB and SGD, on an 80 / 20 split.
* **DL** uses sliding windows of `seq_len` frames (default 30, stride 5), labelled by the last
  frame: LSTM (2 × 128), BiLSTM, 1D-CNN and ST-GCN (3 spatial-temporal graph blocks on the
  skeleton graph of [layouts.py](upfall_bench/skeleton/layouts.py)). Adam (lr 1e-3) with
  class-weighted cross-entropy.
* **Labels** (`skeleton.label`): `class` uses the dataset label; `fall` is 1 when the frame's
  class is in `fall_classes`; `impact` applies the paper's semi-automatic rule: inside each fall,
  frames within ±500 ms of the acceleration-magnitude peak are impacts, if that peak exceeds 2 g.
* **Skeleton sources**: native from the UP-Fall release; `extract-pose` with `torchvision`
  Keypoint R-CNN (COCO-17, 2D; `pose.layout: body12` drops the face joints); or `extract-pose`
  with `mediapipe` BlazePose (33 joints, 3D, as in the paper; needs `pip install mediapipe`).

---

## 13. Deviations from the papers

| topic | paper | here | how to change |
|---|---|---|---|
| feature-selection tools | Weka CfsSubsetEval, CorrelationAttributeEval, ReliefF, ClassifierAttributeEval (ZeroR, DecisionTable) | Python equivalents (§12.3); `mutual_info` replaces ZeroR, which gives every single attribute the same score | `selection.methods` |
| choice of k | read from the curve by eye | smallest k within `tolerance` of the best | `selection.tolerance`, `selection.fixed_k` |
| selection sample | all windows | stratified subsample of 3000 windows | `selection.max_samples` |
| feature formulas | named only | definitions in §12.2 | `arc/features.py` |
| optical flow size | 640 × 480 | computed at 160 × 120, then 20 × 20 | `flow.work_size: [640, 480]` |
| optical flow parameters | Horn–Schunck, parameters not given | α = 1, 10 iterations (MATLAB `opticalFlowHS` defaults) | `flow.alpha`, `flow.iterations` |
| CNN framework | MATLAB (implied by `sgdm`, L2, "regularization coefficient") | PyTorch, same topology and hyper-parameters | `cnn.*` |
| skeleton ML/DL hyper-parameters | not published (repo code missing) | defaults in §12.7, standardised features | `skeleton.*` |
| UP-Fall "unknown" class 12 | 12 classes in Fig. 16 | whatever labels the dataset has (UP-Fall tag 20 = unknown) | adapter |

---

## 14. Methodological notes for a thesis

* **Window-level random splits are optimistic.** With 50 % overlap, neighbouring windows share
  half their samples, so a random 70/30 split puts near-copies of test windows in training. The
  paper does this, so `upfall2019.yaml` does too, to stay comparable. Also report a
  subject-independent result: FAMS has 19 people, so `loso` works (§4.4).
* **Global feature selection leaks**, because it sees the test windows. `selection.scope:
  per_split` (in `subject_independent.yaml`) re-selects on every training set.
* **The MLP is weak by design.** Table 12 caps it at 10 epochs on unscaled features, so the MLP
  and SVM are often poor (as in the paper, where SVM never wins). Report the paper setting and,
  separately, `-s preprocessing.scaler=standard`.
* **Class imbalance.** Falls are a minority of windows, so accuracy says little for the binary
  task. Look at F1 and sensitivity; for selection on imbalanced data, try
  `-s selection.scoring=f1_macro`.
* **Label semantics differ between datasets.** UP-Fall marks only the fall moment as the fall
  activity. UR Fall's fall class here is *lying after the fall*. FAMS labels the whole activity
  segment (~10–20 s). Write down which definition you use.
* **Camera coverage differs from IMU coverage** in FAMS (§15), so CAM and SKELETON results cover
  fewer people than IMU results.
* **Synchronisation.** Session videos are mapped with `started_at_ms` from `cameras.json`. If you
  measure a constant camera offset (e.g. with the flash), shift `t0_ms` in the adapter.

---

## 15. Known issues and troubleshooting

- **Pose extraction crashes on the GPU (exit code 139)** at random points. The cause is not
  found yet; the same code runs without crashing on the CPU, which is far too slow to use. Poses
  are saved in chunks of 64 frames and the pipeline retries, so each crash loses at most one
  chunk. To help find the cause, look at
  `grep -A20 "Fatal Python error" ~/upfall/logs/pose_fams.log | head -40`.
- **Unreadable cache files** left by an interrupted run (flow, poses, window tables) are detected
  and rebuilt automatically.
- **`started with a different configuration`**: delete that results folder or use a new `--out`
  (§10).
- **`no stream matches ['CAM']` / `['SKELETON']`**: run `extract-flow` / `extract-pose` first.
- **`no usable features`**: the streams of that experiment are missing for most windows, so it's
  skipped and the run continues.
- **FAMS cameras:** SF16 has no camera start times and `rabil_fix` has no videos.
- **UR Fall binary task:** few windows count as falls (33 at 1 s, 12 at 2 s, 6 at 3 s), so the
  binary scores, especially at 3 s, are unreliable.
- **Laptop slow or crashing**: lower `resources.n_jobs`, use `-s resources.device=cpu`, and never
  start several commands in parallel.
- **`ffmpeg not found`**: `pip install imageio-ffmpeg`. OpenCV isn't used to read video because VP9
  WebM files from browser recorders have no frame index.
- **No progress shown**: the scripts set `PYTHONUNBUFFERED=1`; if you pipe a command through
  `tee` yourself, add it too.
- **Shared server:** the home disk is shared and nearly full, and other users run GPU jobs. Keep
  `df -h ~` above 10 GB free, and don't run two pipelines at once.
- **MLP "did not converge" warnings**: expected (`max_iter=10` from Table 12) and silenced in `run`.

---

## 16. Project structure

```
UP-Fall-Algorithm/
├── README.md
├── requirements.txt
├── configs/
│   ├── benchmarks/   defaults.yaml · upfall2019.yaml · quick.yaml · subject_independent.yaml · koffi2024_skeleton.yaml
│   └── datasets/     urfall.yaml · fams.yaml · ra_pak_edy.yaml · upfall.yaml · upfall_skeleton.yaml · synthetic.yaml · _template.yaml
├── scripts/
│   ├── run_server.sh      the whole UR Fall + FAMS pipeline, resumable
│   └── run_resumable.sh   one benchmark, restarted after a crash
├── upfall_bench/
│   ├── cli.py            commands (python -m upfall_bench …)
│   ├── config.py         YAML loading / merging / extends / --set / resources
│   ├── core.py           Recording, Stream, VideoSource, sample-and-hold
│   ├── video.py          ffmpeg / image-zip frame reader
│   ├── datasets/         ONE FILE PER DATASET  (+ _template.py, base.py with the registry)
│   ├── derive/           flow.py (Horn–Schunck), pose.py (Keypoint R-CNN / MediaPipe), impact.py (2 g labels)
│   ├── arc/              windowing · features · selection · models · evaluation · runner · report
│   ├── skeleton/         bench.py (ML + DL), dl.py (LSTM, BiLSTM, CNN, ST-GCN), layouts.py (joint graphs)
│   └── vision_cnn.py     Sec. 5.6 CNN
├── tests/test_smoke.py   python -m pytest tests -q
├── data/      (git-ignored) raw datasets
├── cache/     (git-ignored) derived data, safe to delete
└── results/   (git-ignored) runs
```

---

## 17. References

```bibtex
@article{martinez2019upfall,
  title   = {UP-Fall Detection Dataset: A Multimodal Approach},
  author  = {Mart{\'i}nez-Villase{\~n}or, Lourdes and Ponce, Hiram and Brieva, Jorge and Moya-Albor, Ernesto and N{\'u}{\~n}ez-Mart{\'i}nez, Jos{\'e} and Pe{\~n}afort-Asturiano, Carlos},
  journal = {Sensors}, volume = {19}, number = {9}, pages = {1988}, year = {2019},
  doi     = {10.3390/s19091988}
}
@article{koffi2024improved,
  title  = {An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection},
  author = {Koffi, Tresor Y. and Mourchid, Youssef and Hindawi, Mohammed and Dupuis, Yohan},
  year   = {2024}, note = {https://zenodo.org/records/12773013}
}
@article{kwolek2014urfall,
  title   = {Human fall detection on embedded platform using depth maps and wireless accelerometer},
  author  = {Kwolek, Bogdan and Kepski, Michal},
  journal = {Computer Methods and Programs in Biomedicine}, volume = {117}, number = {3}, pages = {489--501}, year = {2014}
}
@article{horn1981,
  title = {Determining optical flow}, author = {Horn, Berthold K. P. and Schunck, Brian G.},
  journal = {Artificial Intelligence}, volume = {17}, pages = {185--203}, year = {1981}
}
@inproceedings{yan2018stgcn,
  title = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author = {Yan, Sijie and Xiong, Yuanjun and Lin, Dahua}, booktitle = {AAAI}, year = {2018}
}
```

The UR Fall dataset is licensed CC BY-NC-SA 4.0 (non-commercial academic use).
