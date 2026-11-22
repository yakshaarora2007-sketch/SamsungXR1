# Project File and Command Guide

This guide describes the files, folders, commands, and generated outputs in this
EuRoC MAV IMU evaluation project. It follows the current implementation; the
older `PROJECT_SUMMARY.md` may not reflect every implementation detail.

## What the project does

The generator selects a continuous 15-second interval from a EuRoC flight
sequence. It saves the clean IMU and ground-truth data, then generates three
evaluation tiers from that same interval:

- **T1 — checkpoints:** selects 10 distinct IMU timestamps and produces pose
  checkpoints from ground truth.
- **T2 — blackout:** removes IMU observations from a random onset between 7 and
  10 seconds through the end of the interval. Ground truth remains available
  for scoring.
- **T3 — sensor degradation:** modifies a random 2-second IMU window beginning
  between 5 and 7 seconds. Data outside the window is retained unchanged.

## Commands

Run commands from the project root. To use the included virtual environment,
replace `python` with `.venv/bin/python`.

| Command | Action |
|---|---|
| `python generate_samples.py --help` | Shows CLI options and exits without generating samples. |
| `python generate_samples.py --init` | Deletes the existing `generated_samples/current_interval/`, selects a new interval, saves its clean input files, and generates `run_001`. **Existing runs in the current interval are deleted; there is no archive folder.** The current interval is cleared before the new interval is selected. |
| `python generate_samples.py --rerun` | Requires an active interval and its clean source CSVs. Reuses them and generates the next numbered run with new randomized tier conditions. It does not clear the active interval. |
| `python generate_samples.py --init --sequence V1_01_easy` | Initializes using the named sequence instead of randomly choosing a sequence. `--sequence` affects interval selection during `--init`. |
| `python generate_samples.py --init --seed 42` | Initializes with a master seed for reproducible random selections and tier generation. |
| `python generate_samples.py --init --dataset-dir /path/to/euroc --output-dir /path/to/output` | Uses the specified dataset and output roots instead of the defaults. |
| `python -m unittest discover tests` | Runs the project unit and integration tests. |

`--init` and `--rerun` are mutually exclusive: provide exactly one. The
default dataset directory is `vicon_room1`; the default output directory is
`generated_samples`. Relative paths are resolved from the directory where the
command is run.

### What happens during `--init`

1. Clears `generated_samples/current_interval/`, including its previous runs.
2. Finds a usable sequence and selects a 15-second span where IMU and ground
   truth overlap.
3. Writes the unmodified interval data and metadata.
4. Generates T1, T2, T3, and plots in a temporary staging directory.
5. Validates the staged tier outputs and moves the run into `run_001`.

Because the current interval is cleared first, a later selection or generation
failure does not restore the previous interval.

### What happens during `--rerun`

1. Requires an active interval, `state.json`, and both original CSV files.
2. Reads the saved clean IMU and ground-truth data.
3. Checks that the first IMU timestamp matches the start timestamp in state.
   It does not verify every timestamp or the complete end boundary.
4. Generates the next run in a staging directory, validates it, and commits it
   under the next `run_NNN` name.

## Project files

### Python modules

| File | Purpose |
|---|---|
| `generate_samples.py` | CLI entry point and workflow coordinator. Handles initialization and reruns, calls the tier generators and graph generator, validates staged runs, and commits results. |
| `interval_selector.py` | Finds sequence folders with IMU and ground-truth CSVs, selects a sequence, and slices a 15-second overlapping interval. It prefers `state_groundtruth_estimate0/data.csv` and falls back to `vicon0/data.csv`. |
| `groundtruth_handler.py` | Standardizes ground-truth columns, slices ground truth, and returns poses for requested timestamps. It uses exact timestamp matches when available; if a requested timestamp is absent, it returns the nearest ground-truth row without interpolation. |
| `t1_generator.py` | Chooses 10 distinct IMU timestamps and formats their ground-truth positions and quaternions into T1 checkpoint output. If there are too few exact timestamp matches, it falls back to IMU timestamps and the handler may use the nearest ground-truth row. |
| `t2_generator.py` | Selects a blackout onset between 7 and 10 seconds, writes only IMU observations strictly before onset, and keeps the supplied ground-truth data for scoring. |
| `t3_generator.py` | Selects a start time between 5 and 7 seconds, chooses degradation settings, and invokes the degradation engine for a 2-second window. |
| `degradation.py` | Implements T3 sensor effects: white noise, bias and scale random walks, spikes, stuck readings, dropped samples, burst gaps, and duplicated samples. It preserves rows outside the selected window. |
| `degrade_imu.py` | Standalone legacy CSV degradation command. It ramps degradation across an input recording; it is not the engine used by the current T3 workflow. |
| `graph_generator.py` | Creates the T2 blackout plot and T3 accelerometer and gyroscope plots as PNG images. |
| `state_manager.py` | Manages the active interval directory, state manifest, run numbering, temporary staging directories, and committing a validated run. |
| `validation.py` | Checks required tier files and basic T1/T2/T3 constraints before a run is committed, including that T3 pre- and post-window data are unchanged. |

### Tests

| File | What it checks |
|---|---|
| `tests/test_interval_selector.py` | Sequence discovery and selection of a nominal 15-second interval from the dataset. |
| `tests/test_groundtruth.py` | Exact ground-truth lookup and checkpoint field formatting. |
| `tests/test_degradation.py` | IMU channel identification and unchanged data outside the T3 degradation window. |
| `tests/test_tiers_and_cli.py` | Tier generator outputs and selected T1/T2/T3 constraints using synthetic data. |

### Supporting and documentation files

| File or folder | Purpose |
|---|---|
| `requirements.txt` | Lists Python dependencies: NumPy, pandas, SciPy, and Matplotlib. |
| `.gitignore` | Excludes datasets, virtual environments, Python cache files, model weights, and macOS `.DS_Store` files from Git. |
| `PROJECT_SUMMARY.md` | Existing long-form project overview and command reference. |
| `PROJECT_GUIDE.pdf` | Visual guide with a workflow diagram, command overview, and folder/file summaries. |
| `PROJECT_FILE_AND_COMMAND_GUIDE.md` | This detailed file and command reference. |
| `.git/` | Git repository metadata and version-control history. |
| `.venv/` | Local Python virtual environment and installed packages; not application source code. |
| `__pycache__/` | Python-generated bytecode cache; not needed as source files. |
| `.DS_Store` | macOS Finder metadata; not part of the application. |

## Dataset folders

`vicon_room1/` contains the included EuRoC sequence directories:

- `V1_01_easy/`
- `V1_02_medium/`
- `V1_03_difficult/`

Each usable sequence has a `mav0/` folder. Relevant files are:

| Path | Purpose |
|---|---|
| `mav0/imu0/data.csv` | Timestamped IMU measurements. In EuRoC, gyroscope columns come before accelerometer columns. |
| `mav0/imu0/sensor.yaml` | IMU sensor configuration and noise/extrinsic information. |
| `mav0/state_groundtruth_estimate0/data.csv` | Ground-truth state/pose data preferred by the interval selector when present. |
| `mav0/state_groundtruth_estimate0/sensor.yaml` | Sensor/frame configuration associated with that state data. |
| `mav0/vicon0/data.csv` | Optical tracking pose data, used as the selector's ground-truth fallback if the preferred state CSV is absent. |
| `mav0/vicon0/sensor.yaml` | Vicon sensor and frame configuration. |
| `<sequence>.bag` and `<sequence>.zip` | Packaged or original sequence recordings; the generator reads the CSV files above. |

## Generated output folders and files

The output root defaults to `generated_samples/`. The current layout is:

```text
generated_samples/
└── current_interval/
    ├── original_imu.csv
    ├── original_groundtruth.csv
    ├── base_interval_metadata.json
    ├── state.json
    └── run_NNN/
        ├── T1/
        ├── T2/
        └── T3/
```

There is no `archived_intervals/` directory in the current workflow. A new
`--init` removes the existing `current_interval/` rather than archiving it.

### Active interval files

| File | Contents |
|---|---|
| `original_imu.csv` | Clean IMU rows from the selected interval, with the original sensor columns and timestamps. |
| `original_groundtruth.csv` | Clean ground-truth rows from the same interval. |
| `base_interval_metadata.json` | Sequence name, interval bounds, nominal/actual duration, sample counts, median IMU time step, and seed. |
| `state.json` | Active interval identity and bounds, run counter, creation time, and committed run records. |

### Files in each run

| Path | Contents |
|---|---|
| `T1/checkpoints.csv` | Ten checkpoint timestamps with position and quaternion columns. |
| `T1/metadata.json` | Checkpoint timestamps, relative times, interval bounds, and random seed. |
| `T2/imu_input.csv` | Available IMU samples before blackout onset; no samples at or after onset. |
| `T2/groundtruth.csv` | The ground-truth DataFrame supplied to the tier generator, retained for scoring. |
| `T2/metadata.json` | Blackout onset, prediction-window boundaries, sample counts, and seed. |
| `T2/graphs/imu_blackout.png` | Plot of available sensor magnitude and blackout/prediction regions. |
| `T3/imu_input.csv` | IMU stream with degradation applied inside the selected window. Dropped and duplicated rows can change the window's row count. |
| `T3/groundtruth.csv` | Ground-truth DataFrame supplied to the tier generator, retained for scoring. |
| `T3/metadata.json` | Degradation window, random seed, settings, and event/sample counts. |
| `T3/graphs/accelerometer_degradation.png` | Plot of accelerometer channels with the degradation window marked. |
| `T3/graphs/gyroscope_degradation.png` | Plot of gyroscope channels with the degradation window marked. |

The hidden `.staging_run_NNN/` directory is temporary: generation writes there,
validation runs against it, and a successful run is moved to `run_NNN/`.

## Standalone legacy degradation command

`degrade_imu.py` can be run independently; the main generator does not invoke
it:

```bash
python degrade_imu.py input.csv output.csv
```

It also accepts optional settings such as `--max-level`, `--gamma`,
`--ts-unit`, `--accel-unit`, `--gyro-deg`, and `--meta`. Use
`python degrade_imu.py --help` for the full option list. This script expects a
seven-column timestamp/accelerometer/gyroscope CSV, with a different default
column order from the native EuRoC IMU CSV.
