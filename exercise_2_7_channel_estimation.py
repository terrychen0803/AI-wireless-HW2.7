#!/usr/bin/env python3
"""
Single-file implementation for Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation.

What this script does
---------------------
1. Simulates a SISO-OFDM system with 64 subcarriers.
2. Uses one OFDM pilot symbol with 64 QPSK pilots and one OFDM data symbol with 64-QAM.
3. Implements and compares:
   - DNN-based channel estimation
   - LMMSE channel estimation
4. Evaluates MSE from SNR = 5, 10, ..., 40 dB.
5. Repeats the experiment with and without CP to show the ISI effect.

Notes
-----
- This is a clean, self-contained version designed for coursework use.
- It uses a synthetic Rayleigh multipath channel with exponential power-delay profile.
- The script preserves the same experiment logic as Exercise 2.7, but does not depend on
  the original multi-file project structure.
- If you want curves closer to the reference figure, increase train_samples, test_samples,
  and epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import toeplitz


# -----------------------------
# Configuration and utilities
# -----------------------------

@dataclass
class SimConfig:
    K: int = 64
    cp_len: int = 16
    num_taps: int = 8
    pdp_decay: float = 0.45
    snr_db_list: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40)
    train_samples: int = 60000
    test_samples: int = 4000
    batch_size: int = 512
    epochs: int = 120
    lr: float = 5e-4
    hidden1: int = 1024
    hidden2: int = 512
    seed: int = 7
    output_dir: str = "results_ex2_7"
    device: str = "auto"  # auto / cpu / cuda


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------
# Modulation
# -----------------------------

def qpsk_symbols(batch: int, K: int, rng: np.random.Generator) -> np.ndarray:
    bits_i = rng.integers(0, 2, size=(batch, K))
    bits_q = rng.integers(0, 2, size=(batch, K))
    symbols = (2 * bits_i - 1) + 1j * (2 * bits_q - 1)
    return symbols.astype(np.complex64) / np.sqrt(2.0)


def qam64_symbols(batch: int, K: int, rng: np.random.Generator) -> np.ndarray:
    # Gray-like 64-QAM amplitude set with unit average power after normalization by sqrt(42)
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    idx_i = rng.integers(0, 8, size=(batch, K))
    idx_q = rng.integers(0, 8, size=(batch, K))
    symbols = levels[idx_i] + 1j * levels[idx_q]
    return symbols.astype(np.complex64) / np.sqrt(42.0)


# -----------------------------
# Channel model and OFDM
# -----------------------------

def exponential_pdp(num_taps: int, decay: float) -> np.ndarray:
    p = np.exp(-decay * np.arange(num_taps, dtype=np.float64))
    return p / p.sum()


def sample_channel(batch: int, cfg: SimConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pdp = exponential_pdp(cfg.num_taps, cfg.pdp_decay)
    real = rng.standard_normal((batch, cfg.num_taps))
    imag = rng.standard_normal((batch, cfg.num_taps))
    taps = (real + 1j * imag) * np.sqrt(pdp[None, :] / 2.0)
    H = np.fft.fft(np.pad(taps, ((0, 0), (0, cfg.K - cfg.num_taps))), axis=1, norm="ortho")
    return taps.astype(np.complex64), H.astype(np.complex64)


def add_prefix(symbol_td: np.ndarray, cp_len: int, cp_flag: bool, prev_td: np.ndarray) -> np.ndarray:
    if cp_flag:
        prefix = symbol_td[-cp_len:]
    else:
        prefix = prev_td[-cp_len:]
    return np.concatenate([prefix, symbol_td])


def transmit_one_symbol(
    X_freq: np.ndarray,
    prev_freq: np.ndarray,
    h_taps: np.ndarray,
    snr_db: float,
    cp_flag: bool,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    x_td = np.fft.ifft(X_freq, norm="ortho")
    prev_td = np.fft.ifft(prev_freq, norm="ortho")

    if cp_flag:
        # Standard CP-OFDM
        tx = np.concatenate([x_td[-cfg.cp_len:], x_td])
        start = cfg.cp_len
    else:
        # No CP: only prepend the previous-symbol tail needed to create ISI
        mem = cfg.num_taps - 1
        tx = np.concatenate([prev_td[-mem:], x_td])
        start = mem

    rx_full = np.convolve(tx, h_taps, mode="full")

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_linear
    noise = (
        rng.standard_normal(rx_full.shape) + 1j * rng.standard_normal(rx_full.shape)
    ) * np.sqrt(noise_var / 2.0)
    rx_full = rx_full + noise

    rx_useful_td = rx_full[start : start + cfg.K]
    Y_freq = np.fft.fft(rx_useful_td, norm="ortho")
    return rx_useful_td.astype(np.complex64), Y_freq.astype(np.complex64)


# -----------------------------
# Dataset generation
# -----------------------------

def make_batch(
    batch_size: int,
    snr_db: float,
    cp_flag: bool,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_taps, H_true = sample_channel(batch_size, cfg, rng)

    X_pilot = qpsk_symbols(batch_size, cfg.K, rng)
    X_prev = qam64_symbols(batch_size, cfg.K, rng)
    _X_data = qam64_symbols(batch_size, cfg.K, rng)  # included to reflect the system setting

    Y_pilot = np.zeros((batch_size, cfg.K), dtype=np.complex64)
    for i in range(batch_size):
        _, Yf = transmit_one_symbol(X_pilot[i], X_prev[i], h_taps[i], snr_db, cp_flag, cfg, rng)
        Y_pilot[i] = Yf

    snr_feat = np.full((batch_size, 1), snr_db / 40.0, dtype=np.float32)

    inputs = np.concatenate(
    [
        Y_pilot.real,
        Y_pilot.imag,
        X_pilot.real,
        X_pilot.imag,
        snr_feat,
    ],
    axis=1,
    ).astype(np.float32)

    labels = np.concatenate([H_true.real, H_true.imag], axis=1).astype(np.float32)
    return inputs, labels, H_true, Y_pilot, X_pilot


# -----------------------------
# LMMSE estimator
# -----------------------------

def channel_covariance_matrix(cfg: SimConfig) -> np.ndarray:
    pdp = exponential_pdp(cfg.num_taps, cfg.pdp_decay)
    delta = np.arange(cfg.K)
    r = np.zeros(cfg.K, dtype=np.complex128)
    tap_idx = np.arange(cfg.num_taps)
    for d in delta:
        r[d] = np.sum(pdp * np.exp(-1j * 2.0 * np.pi * d * tap_idx / cfg.K))
    R = toeplitz(r)
    return R


def lmmse_estimate(Y_pilot: np.ndarray, X_pilot: np.ndarray, snr_db: float, cfg: SimConfig) -> np.ndarray:
    # Because FFT/IFFT use norm="ortho", the CP-OFDM frequency model is:
    # Y = sqrt(K) * X * H + Z
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Correct LS estimate
    H_ls = Y_pilot / (np.sqrt(cfg.K) * X_pilot)

    # Effective noise variance after LS normalization is reduced by K
    R = channel_covariance_matrix(cfg)
    W = R @ np.linalg.inv(R + (1.0 / (cfg.K * snr_linear)) * np.eye(cfg.K, dtype=np.complex128))

    H_mmse = (W @ H_ls.T).T
    return H_mmse.astype(np.complex64)


# -----------------------------
# DNN estimator
# -----------------------------

class ChannelEstimatorMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dnn_joint(
    cp_flag: bool,
    cfg: SimConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> ChannelEstimatorMLP:
    model = ChannelEstimatorMLP(4 * cfg.K + 1, 2 * cfg.K, cfg.hidden1, cfg.hidden2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    steps_per_epoch = math.ceil(cfg.train_samples / cfg.batch_size)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for _ in range(steps_per_epoch):
            # 每個 batch 隨機挑一個 SNR
            snr_db = float(rng.choice(cfg.snr_db_list))

            x_np, y_np, _, _, _ = make_batch(cfg.batch_size, snr_db, cp_flag, cfg, rng)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / steps_per_epoch
        print(f"CP={'on' if cp_flag else 'off'} | epoch {epoch + 1:>3}/{cfg.epochs} | train loss={avg_loss:.6e}")

    return model


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_dnn(
    model: ChannelEstimatorMLP,
    snr_db: float,
    cp_flag: bool,
    cfg: SimConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.eval()
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        remaining = cfg.test_samples
        while remaining > 0:
            bs = min(cfg.batch_size, remaining)
            x_np, _, H_true, _, _ = make_batch(bs, snr_db, cp_flag, cfg, rng)
            x = torch.from_numpy(x_np).to(device)
            pred = model(x).cpu().numpy()

            H_hat = pred[:, : cfg.K] + 1j * pred[:, cfg.K :]
            mse_sum += np.sum(np.mean(np.abs(H_hat - H_true) ** 2, axis=1))
            count += bs
            remaining -= bs

    return float(mse_sum / count)


def evaluate_lmmse(
    snr_db: float,
    cp_flag: bool,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> float:
    mse_sum = 0.0
    count = 0
    remaining = cfg.test_samples

    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        _, _, H_true, Y_pilot, X_pilot = make_batch(bs, snr_db, cp_flag, cfg, rng)
        H_hat = lmmse_estimate(Y_pilot, X_pilot, snr_db, cfg)
        mse_sum += np.sum(np.mean(np.abs(H_hat - H_true) ** 2, axis=1))
        count += bs
        remaining -= bs

    return float(mse_sum / count)


# -----------------------------
# Plotting and orchestration
# -----------------------------

def plot_results(results: Dict[str, List[float]], cfg: SimConfig, out_dir: Path) -> Path:
    snr_list = list(cfg.snr_db_list)
    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_list, results["dnn_cp"], "o-", label="DNN (with CP)")
    plt.semilogy(snr_list, results["lmmse_cp"], "s-", label="LMMSE (with CP)")
    plt.semilogy(snr_list, results["dnn_no_cp"], "o--", label="DNN (no CP)")
    plt.semilogy(snr_list, results["lmmse_no_cp"], "s--", label="LMMSE (no CP)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.title("SISO-OFDM Channel Estimation (Exercise 2.7)")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    fig_path = out_dir / "figure_2_9_reproduction.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def run_experiment(cfg: SimConfig) -> Dict[str, List[float]]:
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    results: Dict[str, List[float]] = {
        "dnn_cp": [],
        "lmmse_cp": [],
        "dnn_no_cp": [],
        "lmmse_no_cp": [],
    }

    rng_train_cp = np.random.default_rng(cfg.seed + 11)
    rng_test_cp_dnn = np.random.default_rng(cfg.seed + 21)
    rng_test_cp_mmse = np.random.default_rng(cfg.seed + 31)
    rng_train_no_cp = np.random.default_rng(cfg.seed + 41)
    rng_test_no_cp_dnn = np.random.default_rng(cfg.seed + 51)
    rng_test_no_cp_mmse = np.random.default_rng(cfg.seed + 61)

    print("\nTraining joint DNN for WITH-CP ...")
    model_cp = train_dnn_joint(True, cfg, device, rng_train_cp)

    print("\nTraining joint DNN for NO-CP ...")
    model_no_cp = train_dnn_joint(False, cfg, device, rng_train_no_cp)

    for snr_db in cfg.snr_db_list:
        mse_dnn_cp = evaluate_dnn(model_cp, snr_db, True, cfg, device, rng_test_cp_dnn)
        mse_lmmse_cp = evaluate_lmmse(snr_db, True, cfg, rng_test_cp_mmse)

        mse_dnn_no_cp = evaluate_dnn(model_no_cp, snr_db, False, cfg, device, rng_test_no_cp_dnn)
        mse_lmmse_no_cp = evaluate_lmmse(snr_db, False, cfg, rng_test_no_cp_mmse)

        results["dnn_cp"].append(mse_dnn_cp)
        results["lmmse_cp"].append(mse_lmmse_cp)
        results["dnn_no_cp"].append(mse_dnn_no_cp)
        results["lmmse_no_cp"].append(mse_lmmse_no_cp)

        print(
            f"SNR={snr_db:>2} dB | "
            f"DNN_CP={mse_dnn_cp:.6e} | LMMSE_CP={mse_lmmse_cp:.6e} | "
            f"DNN_noCP={mse_dnn_no_cp:.6e} | LMMSE_noCP={mse_lmmse_no_cp:.6e}"
        )

    fig_path = plot_results(results, cfg, out_dir)
    print(f"\nSaved figure to: {fig_path}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2)

    return results


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Single-file Exercise 2.7 SISO-OFDM channel estimation")
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--test-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-taps", type=int, default=8)
    parser.add_argument("--pdp-decay", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="results_ex2_7")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    return SimConfig(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_taps=args.num_taps,
        pdp_decay=args.pdp_decay,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_experiment(cfg)
