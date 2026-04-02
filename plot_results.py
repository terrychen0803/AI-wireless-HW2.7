import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

DEFAULT_SNRS = np.arange(5, 45, 5)


def load_mse_mat(mat_path: Path):
    """Load one .mat file saved by main.py and return a 1-D numpy array."""
    data = sio.loadmat(mat_path)
    # main.py saves {savefile: np.array(MSE_F)}
    candidate_keys = [k for k in data.keys() if not k.startswith('__')]
    if not candidate_keys:
        raise ValueError(f'No valid variable found in {mat_path}')

    # Prefer the variable name matching the file stem.
    key = mat_path.stem if mat_path.stem in data else candidate_keys[0]
    arr = np.asarray(data[key]).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(float)


def maybe_to_db(arr: np.ndarray):
    """If values look linear-scale MSE, convert to dB; otherwise keep as-is."""
    arr = np.asarray(arr, dtype=float)
    # Existing saved outputs from main.py are linear NMSE averages.
    # Convert positive values to dB.
    if np.all(arr > 0):
        return 10 * np.log10(arr)
    return arr


def add_curve(ax, folder: Path, stem: str, label: str, linestyle: str = '-', marker: str = 'o'):
    mat_path = folder / f'{stem}.mat'
    if not mat_path.exists():
        print(f'[SKIP] {mat_path.name} not found')
        return False

    y = load_mse_mat(mat_path)
    y_db = maybe_to_db(y)
    x = DEFAULT_SNRS[: len(y_db)]
    ax.plot(x, y_db, linestyle=linestyle, marker=marker, label=label)
    print(f'[LOAD] {mat_path.name}: {y_db}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot channel-estimation MSE results from .mat files.')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing MSE_*.mat files')
    parser.add_argument('--save', type=str, default='figure_2_9_reproduced.png', help='Output image filename')
    parser.add_argument('--show', action='store_true', help='Show the figure window')
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise FileNotFoundError(f'Folder not found: {folder}')

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted_any = False
    plotted_any |= add_curve(ax, folder, 'MSE_dnn_4QAM', 'DNN (with CP)', '-', 'o')
    plotted_any |= add_curve(ax, folder, 'MSE_mmse_4QAM', 'LMMSE (with CP)', '-', 's')
    plotted_any |= add_curve(ax, folder, 'MSE_ls_4QAM', 'LS (with CP)', '-', '^')

    plotted_any |= add_curve(ax, folder, 'MSE_dnn_4QAM_CP_FREE', 'DNN (without CP)', '--', 'o')
    plotted_any |= add_curve(ax, folder, 'MSE_mmse_4QAM_CP_FREE', 'LMMSE (without CP)', '--', 's')
    plotted_any |= add_curve(ax, folder, 'MSE_ls_4QAM_CP_FREE', 'LS (without CP)', '--', '^')

    if not plotted_any:
        raise FileNotFoundError(
            'No matching MSE_*.mat files were found. '\
            'Please run main.py first to generate results.'
        )

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    ax.set_title('SISO-OFDM Channel Estimation Performance')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    save_path = folder / args.save
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[SAVE] {save_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
