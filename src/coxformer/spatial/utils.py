# ====== Basic ======
import os
import pickle
import gzip
import tempfile

# ====== Numeric / Data ======
import numpy as np
import pandas as pd

# ====== Deep Learning ======
import torch

# ====== Image ======
from PIL import Image

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed: int = 42, deterministic: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_paths(base_path, pattern, task):
    """Return dataset file paths according to pattern."""
    return {
        "spatial":   os.path.join(base_path, "cnts.tsv"),
        "hist":      os.path.join(base_path, "image_embedding.pkl"),
        "hist_pixel":os.path.join(base_path, "embeddings-hist.pickle"),
        "genes_txt": os.path.join(base_path, "gene-names.txt"),
        "locs":      os.path.join(base_path, "locs.tsv"),
        "slide_num":  os.path.join(base_path, "slide_num.txt"),
        "genes_train": os.path.join(base_path, "genes_train.npy"),
        "genes_test":  os.path.join(base_path, "genes_test.npy"),
    }




def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x

def save_atomic(obj, path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".pkl.gz")
    os.close(fd)
    try:
        with gzip.open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def convert_to_dataframe(y_true: np.ndarray, y_pred: np.ndarray, genes: list) -> tuple:
    true_df = pd.DataFrame(y_true.T, columns=genes)
    pred_df = pd.DataFrame(y_pred.T, columns=genes)
    return true_df, pred_df