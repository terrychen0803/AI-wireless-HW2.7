# Exercise 2.7 — SISO-OFDM Channel Estimation

This project is a **single-file Python implementation** for reproducing the main experiment in **Exercise 2.7**: comparing **DNN-based** and **LMMSE** channel estimators in a **SISO-OFDM** system, under both **with-CP** and **without-CP** settings.

The script evaluates channel estimation performance in terms of **MSE** for SNR values from **5 dB to 40 dB**.

---

## File

- `exercise_2_7_channel_estimation.py`

This is a self-contained script that includes:
- OFDM signal generation
- QPSK pilot generation
- 64-QAM data generation
- Multipath channel simulation
- CP / no-CP transmission modeling
- LMMSE channel estimation
- DNN-based channel estimation
- MSE evaluation
- Figure plotting and result export

---

## Experiment Overview

The script simulates the following system:

- **SISO-OFDM**
- **64 subcarriers**
- **First OFDM symbol:** 64 QPSK pilot symbols
- **Second OFDM symbol:** 64-QAM data symbols
- **Channel:** synthetic Rayleigh multipath fading channel
- **Metric:** Mean Squared Error (MSE)
- **SNR range:** 5, 10, 15, ..., 40 dB

Two channel estimation methods are compared:

1. **DNN-based channel estimation**
2. **LMMSE channel estimation**

Each method is tested in two scenarios:

- **With CP**
- **Without CP**

The **without-CP** case is used to demonstrate the effect of **inter-symbol interference (ISI)**.

---

## Requirements

Install the required Python packages before running the script.

```bash
pip install numpy scipy matplotlib torch
```

Recommended Python version:

- Python 3.9 or later

---

## How to Run

### Default run

```bash
python exercise_2_7_channel_estimation.py
```

This will:
- train a DNN estimator for each SNR
- evaluate DNN and LMMSE
- run both CP and no-CP experiments
- save the figure and numerical results

## Output

After execution, the script creates an output folder:

```text
results_ex2_7/
```

Inside this folder, the following files are generated:

- `figure_2_9_reproduction.png`  
  Plot of MSE vs. SNR for all four cases:
  - DNN with CP
  - LMMSE with CP
  - DNN without CP
  - LMMSE without CP

- `results.json`  
  Stores the simulation configuration and numerical MSE results.

---

## Main Parameters

The script supports the following command-line arguments:

| Argument | Description | Default |
|---|---|---:|
| `--train-samples` | Number of generated training samples per SNR | `12000` |
| `--test-samples` | Number of generated test samples per SNR | `4000` |
| `--batch-size` | Batch size for training and evaluation | `256` |
| `--epochs` | Training epochs for the DNN | `30` |
| `--lr` | Learning rate | `1e-3` |
| `--num-taps` | Number of channel taps | `8` |
| `--pdp-decay` | Exponential power delay profile decay factor | `0.45` |
| `--seed` | Random seed | `7` |
| `--output-dir` | Output directory | `results_ex2_7` |
| `--device` | Compute device: `auto`, `cpu`, or `cuda` | `auto` |

Example:

```bash
python exercise_2_7_channel_estimation.py \
  --epochs 20 \
  --train-samples 10000 \
  --test-samples 3000 \
  --device cuda
```

---

## Code Structure

The script is organized into the following parts:

### 1. Configuration
Defines simulation parameters in the `SimConfig` dataclass.

### 2. Modulation
Implements:
- QPSK symbol generation for pilots
- 64-QAM symbol generation for data

### 3. Channel Model and OFDM
Implements:
- multipath Rayleigh channel generation
- exponential power delay profile
- OFDM modulation/demodulation
- CP insertion and removal
- no-CP transmission for ISI demonstration

### 4. Dataset Generation
Generates:
- DNN input features: real/imaginary parts of received and transmitted pilots
- labels: real/imaginary parts of the true frequency-domain channel

### 5. LMMSE Estimator
Computes:
- LS estimate from pilots
- covariance-based LMMSE refinement

### 6. DNN Estimator
Uses a multilayer perceptron with:
- input dimension: `4K`
- output dimension: `2K`
- two hidden layers
- ReLU activations

### 7. Evaluation
Computes average MSE over the test set for:
- DNN
- LMMSE

### 8. Plotting and Saving
Generates the final comparison figure and saves the experiment results.

---

## DNN Input and Output Format

For each sample:

### Input
The DNN input is formed by concatenating:
- real part of received pilot vector `Y_pilot`
- imaginary part of received pilot vector `Y_pilot`
- real part of transmitted pilot vector `X_pilot`
- imaginary part of transmitted pilot vector `X_pilot`

So the total input dimension is:

```text
4 × K = 256
```

### Output
The DNN predicts the full frequency-domain channel:
- real part of `H`
- imaginary part of `H`

So the output dimension is:

```text
2 × K = 128
```

---

## Expected Behavior

Typical trends you should observe:

- **With CP:**
  - DNN and LMMSE should have similar performance.
- **Without CP:**
  - LMMSE should degrade more significantly due to ISI.
  - DNN is usually more robust because it learns directly from distorted pilot observations.

If the reproduced curve is noisy or not close enough to the reference figure, try increasing:

- `--train-samples`
- `--test-samples`
- `--epochs`

---

