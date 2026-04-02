# Exercise 2.7 — Data-Driven SISO-OFDM Channel Estimation

This repository reproduces the SISO-OFDM channel estimation experiment in **Section 2.3.1** of the course/book materials on *Wireless Communications and Machine Learning*. The goal is to compare a **DNN-based channel estimator** and a **traditional LMMSE estimator** under two settings:

- **With cyclic prefix (CP)**
- **Without cyclic prefix (CP-free)**

The implemented OFDM system uses:

- **64 subcarriers**
- **1st OFDM symbol:** 64 QPSK pilot symbols
- **2nd OFDM symbol:** 64-QAM data symbols
- **SNR sweep:** 5 dB to 40 dB in 5 dB steps

The repository includes training, evaluation, and plotting scripts, along with saved model checkpoints and example result figures.

---

## 1. Project Objectives

This project focuses on the following tasks:

1. Implement a **DNN-based channel estimator** for SISO-OFDM.
2. Implement a **traditional LMMSE channel estimator**.
3. Compare their mean square error (MSE) performance across different SNR values.
4. Investigate the effect of removing CP, which introduces **inter-symbol interference (ISI)** and **inter-carrier interference (ICI)**.
5. Reproduce the trend of the reference figure in which:
   - with CP, DNN and LMMSE achieve similar performance;
   - without CP, DNN remains relatively robust while LMMSE degrades due to model mismatch.

---

## 2. Repository Structure

```text
exercise_2_7/
├── main.py                         # Main entry point for training/testing
├── plot_results.py                 # Plot MSE curves from saved .mat files
├── README.md                       # Project description
├── figure_2_9_reproduced.png       # Example result plot
├── MSE_dnn_4QAM.mat                # DNN result (with CP)
├── MSE_dnn_4QAM_CP_FREE.mat        # DNN result (without CP)
├── MSE_mmse_4QAM.mat               # LMMSE result (with CP)
├── MSE_mmse_4QAM_CP_FREE.mat       # LMMSE result (without CP)
├── dnn_ce/                         # Saved DNN checkpoints (.npz)
└── tools/
    ├── __init__.py
    ├── networks.py                 # DNN model definition and training
    ├── raputil.py                  # OFDM simulation, LS/MMSE estimation, test utilities
    ├── train.py                    # Save/load trainable variables
    ├── shrinkage.py                # Reserved from original repo structure
    ├── problems.py                 # Reserved from original repo structure
    └── Pilot_64_mu2.txt            # Saved pilot sequence
```

---

## 3. Environment Setup

### 3.1 Recommended Environment

This project was written in Python and uses TensorFlow in **compatibility mode** (`tensorflow.compat.v1`).

Recommended setup:

- Python 3.10 or 3.11
- NumPy
- SciPy
- Matplotlib
- TensorFlow 2.x (using `compat.v1`)

### 3.2 Install Dependencies

Create a virtual environment if desired:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install required packages:

```bash
pip install numpy scipy matplotlib tensorflow
```

> If TensorFlow installation depends on your platform/CPU/GPU, choose the version suitable for your system.

---

## 4. How to Run

The main script is `main.py`. You control the experiment by editing three parameters near the top of the file:

```python
ce_type = 'mmse'   # 'ls', 'mmse', or 'dnn'
test_ce = True     # False: train DNN, True: evaluate estimator
CP_flag = False    # True: with CP, False: CP-free
```

### 4.1 Important Note for DNN Training

Before training the DNN, create the checkpoint folder manually if it does not already exist:

```bash
mkdir dnn_ce
```

On Windows PowerShell:

```powershell
mkdir dnn_ce
```

If `dnn_ce/` does not exist, saving the `.npz` checkpoint files will fail.

---

### 4.2 Run LMMSE (with CP)

Edit `main.py`:

```python
ce_type = 'mmse'
test_ce = True
CP_flag = True
```

Run:

```bash
python main.py
```

This generates:

- `MSE_mmse_4QAM.mat`

---

### 4.3 Run LMMSE (without CP)

Edit `main.py`:

```python
ce_type = 'mmse'
test_ce = True
CP_flag = False
```

Run:

```bash
python main.py
```

This generates:

- `MSE_mmse_4QAM_CP_FREE.mat`

---

### 4.4 Train DNN (with CP)

Edit `main.py`:

```python
ce_type = 'dnn'
test_ce = False
CP_flag = True
```

Run:

```bash
python main.py
```

This trains the DNN for each SNR point and saves checkpoints under `dnn_ce/`.

---

### 4.5 Evaluate DNN (with CP)

Edit `main.py`:

```python
ce_type = 'dnn'
test_ce = True
CP_flag = True
```

Run:

```bash
python main.py
```

This generates:

- `MSE_dnn_4QAM.mat`

---

### 4.6 Train and Evaluate DNN (without CP)

For training:

```python
ce_type = 'dnn'
test_ce = False
CP_flag = False
```

Run:

```bash
python main.py
```

Then evaluate using:

```python
ce_type = 'dnn'
test_ce = True
CP_flag = False
```

Run:

```bash
python main.py
```

This generates:

- `MSE_dnn_4QAM_CP_FREE.mat`

---

## 5. Plotting the Results

Use `plot_results.py` to visualize the saved `.mat` files:

```bash
python plot_results.py --dir . --show
```

To save the figure without displaying it:

```bash
python plot_results.py --dir .
```

Default output:

- `figure_2_9_reproduced.png`

The plot script automatically searches for the following files if they exist:

- `MSE_dnn_4QAM.mat`
- `MSE_dnn_4QAM_CP_FREE.mat`
- `MSE_mmse_4QAM.mat`
- `MSE_mmse_4QAM_CP_FREE.mat`
- `MSE_ls_4QAM.mat`
- `MSE_ls_4QAM_CP_FREE.mat`

---


## 6. Code Explanation

### `main.py`

This is the main experiment script.

Responsibilities:

- defines the SNR sweep (`5, 10, ..., 40 dB`)
- chooses the estimator type (`ls`, `mmse`, `dnn`)
- chooses CP or CP-free configuration
- trains the DNN when `test_ce = False`
- evaluates MSE when `test_ce = True`
- saves results to `.mat` files for plotting

### `tools/networks.py`

This file defines the DNN channel estimator.

Model summary:

- input: real/imaginary parts of received pilot and transmitted pilot
- architecture: fully connected DNN with two hidden layers
- activations: ReLU
- output: real/imaginary parts of the estimated 64-subcarrier channel
- loss: mean square error
- optimizer: Adam

### `tools/raputil.py`

This is the core communication-system utility file.

Key functions:

- modulation / demodulation for QPSK, 16-QAM, and 64-QAM
- OFDM IFFT/FFT processing
- CP insertion and removal
- channel convolution and AWGN addition
- LS channel estimation
- LMMSE channel estimation
- sample generation for DNN training and validation
- MSE evaluation for different channel estimators

This file also contains the CP-free simulation logic.

### `tools/train.py`

Provides helper functions to:

- save DNN weights to `.npz`
- restore saved model parameters

### `plot_results.py`

Reads the saved `.mat` files and converts linear-scale MSE into dB before plotting.

---

## 7. Dataset Handling

The code first attempts to load the original channel datasets:

- `tools/channel_train.npy`
- `tools/channel_test.npy`

If these files are not found, the implementation automatically falls back to **synthetic Rayleigh fading channels** generated inside `tools/raputil.py`.

This allows the full pipeline to run even when the original dataset files are unavailable.

---

## 8. Experimental Results and Analysis

![image](https://github.com/terrychen0803/AI-wireless-HW2.7/blob/main/figure_2_9_reproduced.png)

The final results included in this repository show the following trends.

### 8.1 With CP

Representative frequency-domain MSE (dB):

| SNR (dB) | DNN | LMMSE |
|---|---:|---:|
| 5  | -12.57 | -13.37 |
| 10 | -16.64 | -17.62 |
| 15 | -21.10 | -22.21 |
| 20 | -25.58 | -27.02 |
| 25 | -29.15 | -31.92 |
| 30 | -32.75 | -36.91 |
| 35 | -36.18 | -41.88 |
| 40 | -37.13 | -46.91 |

**Observation:**

- Both methods improve as SNR increases.
- LMMSE performs very strongly when CP is present because its underlying linear model is valid.
- The DNN also achieves good performance and follows the same overall trend.

### 8.2 Without CP

Representative frequency-domain MSE (dB):

| SNR (dB) | DNN | LMMSE |
|---|---:|---:|
| 5  | -12.01 | -11.91 |
| 10 | -15.80 | -14.46 |
| 15 | -19.89 | -15.94 |
| 20 | -24.16 | -16.55 |
| 25 | -27.87 | -16.45 |
| 30 | -31.31 | -16.64 |
| 35 | -33.43 | -16.88 |
| 40 | -34.31 | -16.77 |

**Observation:**

- The DNN still improves as SNR increases, although it is slightly worse than the with-CP case.
- LMMSE quickly reaches a performance floor and no longer improves much at high SNR.
- This is a typical sign of **model mismatch** caused by CP removal, ISI, and ICI.

### 8.3 Overall Conclusion

The main conclusions are:

1. **With CP**, both DNN and LMMSE work well because the OFDM frequency-domain model remains valid.
2. **Without CP**, the channel estimation problem becomes harder because the received pilot no longer follows the simple per-subcarrier model.
3. The **DNN-based estimator is more robust** in the CP-free case because it can learn a nonlinear mapping directly from distorted pilot observations.
4. The **LMMSE estimator is more sensitive to model mismatch**, leading to a clear MSE floor in the CP-free setting.

These results are consistent with the intended qualitative behavior of the reference experiment.

---

