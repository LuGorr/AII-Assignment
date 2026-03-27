"""
shared_utils.py
===============
Shared data pipeline for the GC detection project.

IMPORTANT — DO NOT MODIFY THIS FILE.
These functions are copied verbatim from the DCGP notebook (gp.ipynb)
to guarantee identical splits, class weights, and data loading across
all three model implementations (CNN-GMM, DeepCGP, RF baseline).

Any change here must be agreed on by the whole team, since it will
change the train/val/test split and make cross-model comparisons invalid.

"""

# ── Imports ────────────────────────────────────────────────────────────────────
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Project data loader ────────────────────────────────────────────────────────
# data.py must be on the Python path (it lives in utils/ in the shared repo).
from utils import data


# ==============================================================================
# 1.  REPRODUCIBILITY
# ==============================================================================

def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds to `seed` for full reproducibility.

    Call this ONCE at the very top of your notebook / script,
    before any data loading or model construction.

    Mirrors the seed block in gp.ipynb:
        torch.manual_seed(42)
        np.random.seed(42)
    Extended here with CUDA and Python seeds so GPU runs are also
    deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes cuDNN deterministic.  Slight speed cost — acceptable for research.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[shared_utils] All seeds set to {seed}.")


# ==============================================================================
# 2.  DATA LOADING
# ==============================================================================

def load_all_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load FCC + VCC imaging data, drop NaN frames, and compute class weights.

    Returns
    -------
    all_frames : pd.DataFrame
        Merged, NaN-cleaned dataset (~85 k rows).
        Columns include 'frame' (np.ndarray, shape 2×20×20) and 'y' (bool).
    fcc_df : pd.DataFrame
        FCC-only subset with NaN frames removed (used for cross-cluster transfer).
    vcc_df : pd.DataFrame
        VCC-only subset with NaN frames removed (used for cross-cluster transfer).

    Notes
    -----
    Copied verbatim from gp.ipynb §2 "Data loading" cell.
    The NaN filter `~frame.apply(lambda x: np.isnan(x).any())` is identical
    in both the merged and per-cluster dataframes.
    """
    FCC, VCC = data.prepare_data()

    # ── merged dataset ──────────────────────────────────────────────────────
    all_frames = pd.concat((FCC, VCC), ignore_index=True)
    all_frames = all_frames[
        ~all_frames.frame.apply(lambda x: np.isnan(x).any())
    ]

    # ── per-cluster (for transfer experiments) ──────────────────────────────
    fcc_df = FCC[~FCC.frame.apply(lambda x: np.isnan(x).any())]
    vcc_df = VCC[~VCC.frame.apply(lambda x: np.isnan(x).any())]

    return all_frames, fcc_df, vcc_df


def compute_class_weights(
    all_frames: pd.DataFrame,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Compute inverse-frequency class weights.

    Formula (verbatim from gp.ipynb):
        w_pos = total / (2 * n_pos)
        w_neg = total / (2 * n_neg)
        CLASS_WEIGHTS = torch.tensor([w_neg, w_pos], ...)

    Index 0 = non-GC (negative class, label=False/0)
    Index 1 = GC     (positive class, label=True/1)

    Returns
    -------
    class_weights : torch.Tensor, shape (2,), on `device`
    stats : dict  with keys total, n_pos, n_neg, w_pos, w_neg
    """
    counts = all_frames['y'].value_counts()
    n_neg  = int(counts[False])
    n_pos  = int(counts[True])
    total  = n_neg + n_pos

    w_pos = total / (2 * n_pos)
    w_neg = total / (2 * n_neg)

    class_weights = torch.tensor(
        [w_neg, w_pos], dtype=torch.float32
    ).to(device)

    stats = dict(total=total, n_pos=n_pos, n_neg=n_neg,
                 w_pos=w_pos, w_neg=w_neg)
    return class_weights, stats


# ==============================================================================
# 3.  TRAIN / VAL / TEST SPLIT
# ==============================================================================

def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified random split into train / val / test.

    Copied VERBATIM from gp.ipynb §2.1:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        return df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]

    Default split: 70 % train / 10 % val / 20 % test.
    With seed=42 this produces identical row assignments in every run
    and in every notebook that calls this function.

    Parameters
    ----------
    df         : full dataframe to split
    train_frac : fraction for training
    val_frac   : fraction for validation (remainder goes to test)
    seed       : random state — ALWAYS keep at 42 for cross-model consistency

    Returns
    -------
    train_df, val_df, test_df
    """
    df      = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n       = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return df[:n_train], df[n_train:n_train + n_val], df[n_train + n_val:]


# ==============================================================================
# 4.  DATASET
# ==============================================================================

class FrameDataset(Dataset):
    """
    Two-channel (g, z) image dataset for GC classification.

    Copied VERBATIM from gp.ipynb §3.
    Frames are stored as float32 arrays of shape (2, 20, 20) — channels first.
    Labels are int64 (0 = non-GC, 1 = GC).

    Parameters
    ----------
    df        : DataFrame with columns 'frame' (np.ndarray) and 'y' (bool/int)
    transform : optional callable applied independently to each channel
                (same interface as in the DCGP notebook)
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # Stack all frames: (N, 2, 20, 20) float32
        self.frames    = np.stack(df.frame.values).astype('float32')
        self.labels    = np.array(df.y, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.frames[idx]                          # (2, 20, 20)
        if self.transform:
            ch0 = self.transform(img[0])
            ch1 = self.transform(img[1])
            img = np.stack([ch0, ch1])
        return torch.tensor(img).float(), torch.tensor(self.labels[idx])


# ==============================================================================
# 5.  DATALOADERS
# ==============================================================================

def make_loader(
    df:          pd.DataFrame,
    batch_size:  int  = 64,
    shuffle:     bool = True,
    num_workers: int  = 4,
    transform         = None,
) -> DataLoader:
    """
    Wrap a DataFrame in a FrameDataset and return a DataLoader.

    Copied VERBATIM from gp.ipynb §3.
    Default batch_size=64, pin_memory=True.

    Parameters
    ----------
    df          : DataFrame with 'frame' and 'y' columns
    batch_size  : mini-batch size (default 64 — matches DCGP notebook)
    shuffle     : True for train, False for val/test
    num_workers : number of DataLoader workers
    transform   : optional per-channel transform (passed to FrameDataset)
    """
    return DataLoader(
        FrameDataset(df, transform=transform),
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
    )


# ==============================================================================
# 6.  VERIFICATION HELPER
# ==============================================================================

def verify_setup(
    all_frames:    pd.DataFrame,
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    test_df:       pd.DataFrame,
    class_weights: torch.Tensor,
    stats:         dict,
) -> None:
    """
    Run a set of assertions to confirm the data pipeline is consistent
    with the DCGP notebook.  Call this once after loading data.

    Checks
    ------
    1. Frame shape is (2, 20, 20)
    2. Split sizes sum to total
    3. Split fractions are within 0.5% of 70/10/20
    4. Class weight tensor has shape (2,) and both values are positive
    5. GC count is in the expected range (~18k)
    6. non-GC count is in the expected range (~67k)
    7. No overlap between train/val/test (index check)
    8. One sample forward-pass through FrameDataset returns correct dtypes
    """
    print("=" * 60)
    print("SHARED_UTILS — VERIFICATION")
    print("=" * 60)

    # 1. Frame shape
    sample_frame = all_frames.frame.iloc[0]
    assert sample_frame.shape == (2, 20, 20), (
        f"Expected frame shape (2, 20, 20), got {sample_frame.shape}"
    )
    print(f"  [OK] Frame shape          : {sample_frame.shape}")

    # 2. Split sizes
    total    = stats['total']
    n_splits = len(train_df) + len(val_df) + len(test_df)
    assert n_splits == total, (
        f"Split sizes {n_splits} != total {total}"
    )
    print(f"  [OK] Split sizes sum       : {len(train_df):,} + "
          f"{len(val_df):,} + {len(test_df):,} = {n_splits:,}")

    # 3. Split fractions
    tol = 0.005
    train_frac_actual = len(train_df) / total
    val_frac_actual   = len(val_df)   / total
    test_frac_actual  = len(test_df)  / total
    assert abs(train_frac_actual - 0.70) < tol, \
        f"Train fraction {train_frac_actual:.4f} not close to 0.70"
    assert abs(val_frac_actual   - 0.10) < tol, \
        f"Val fraction {val_frac_actual:.4f} not close to 0.10"
    print(f"  [OK] Split fractions       : train={train_frac_actual:.3f}  "
          f"val={val_frac_actual:.3f}  test={test_frac_actual:.3f}")

    # 4. Class weights
    assert class_weights.shape == (2,), \
        f"CLASS_WEIGHTS shape {class_weights.shape} != (2,)"
    assert (class_weights > 0).all(), \
        "CLASS_WEIGHTS contains non-positive values"
    print(f"  [OK] Class weights         : non-GC={class_weights[0]:.4f}, "
          f"GC={class_weights[1]:.4f}")

    # 5-6. Class counts (rough sanity — exact counts depend on NaN removal)
    n_pos, n_neg = stats['n_pos'], stats['n_neg']
    assert 15_000 < n_pos < 22_000, \
        f"GC count {n_pos} outside expected range [15k, 22k]"
    assert 60_000 < n_neg < 80_000, \
        f"non-GC count {n_neg} outside expected range [60k, 80k]"
    print(f"  [OK] Class counts          : GC={n_pos:,}  non-GC={n_neg:,}  "
          f"ratio=1:{n_neg/n_pos:.1f}")

    # 7. No overlap (use the integer positional index after reset_index)
    train_idx = set(train_df.index.tolist())
    val_idx   = set(val_df.index.tolist())
    test_idx  = set(test_df.index.tolist())
    assert len(train_idx & val_idx)  == 0, "Train/Val overlap detected!"
    assert len(train_idx & test_idx) == 0, "Train/Test overlap detected!"
    assert len(val_idx   & test_idx) == 0, "Val/Test overlap detected!"
    print("  [OK] No train/val/test overlap")

    # 8. Dataset forward pass
    ds     = FrameDataset(train_df)
    x, y   = ds[0]
    assert x.dtype == torch.float32, f"Frame dtype {x.dtype} != float32"
    assert y.dtype == torch.int64,   f"Label dtype {y.dtype} != int64"
    assert x.shape == (2, 20, 20),   f"Tensor shape {x.shape} != (2, 20, 20)"
    assert y.item() in (0, 1),       f"Label {y.item()} not in {{0, 1}}"
    print(f"  [OK] Dataset __getitem__   : x.shape={tuple(x.shape)}  "
          f"x.dtype={x.dtype}  y={y.item()}")

    print("=" * 60)
    print("All checks passed — pipeline is consistent with gp.ipynb.")
    print("=" * 60)
