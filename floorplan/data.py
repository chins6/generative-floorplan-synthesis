"""Dataset access: official split files, preprocessing, caching and corruption.

Preprocessing (unchanged from the mid-review): binarise the full-resolution
F1_original.png at grey level < 200, then nearest-neighbour resize to 32x32.
The three splits are cached as .npy so the dataset itself is only needed once.
"""
import re
import time

import numpy as np

from . import config as C


def find_dataset_dir():
    """Return the CubiCasa5K folder that holds train.txt, or None if absent."""
    for layout in C.DATASET_LAYOUTS:
        if (layout / "train.txt").is_file():
            return layout
    return None


def read_split(dataset_dir, name):
    """List the sample folders of one official split file (train/val/test)."""
    with open(dataset_dir / f"{name}.txt", "r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    return [dataset_dir / s.strip("/") for s in ids]


def load_wall_map(png_path, side=C.SIDE):
    """One floorplan PNG -> flat float32 array of 0/1 of length side*side."""
    from PIL import Image  # imported lazily: not needed when the cache exists
    image = Image.open(png_path).convert("L")
    binary = image.point(lambda v: v < C.WALL_THRESHOLD, mode="1")   # binarise first
    small = binary.resize((side, side), Image.Resampling.NEAREST)    # then shrink
    return np.asarray(small, dtype=np.float32).reshape(-1)


_WALL_RE = re.compile(r'<g\b[^>]*\bclass="Wall(?:\s[^"]*)?"[^>]*>\s*<polygon\b[^>]*\bpoints="([^"]+)"')


def load_svg_walls(svg_path, side=C.SIDE):
    """Official labels: rasterise ONLY the Wall polygons of floor 1 of model.svg.

    1. keep the first floor (the PNG used in stage 1 is F1_original.png too),
    2. crop to the walls' bounding box and PAD to a square (no stretching),
    3. draw the filled polygons at WALL_RASTER x WALL_RASTER,
    4. reduce to side x side by AREA: a cell is wall if >= WALL_MIN_COVER of it is wall.
       Walls are thinner than one 32x32 cell, so a 50% area rule would erase them.
    """
    from PIL import Image, ImageDraw
    text = svg_path.read_text(encoding="utf-8", errors="ignore")
    cut = text.find('class="Floorplan Floor-2"')
    if cut > 0:
        text = text[:cut]
    polys = []
    for pts in _WALL_RE.findall(text):
        xy = np.array([[float(v) for v in p.split(",")] for p in pts.split()], dtype=np.float64)
        if len(xy) >= 3:
            polys.append(xy)
    if not polys:
        return np.zeros(side * side, np.float32)
    allpts = np.concatenate(polys)
    lo, hi = allpts.min(0), allpts.max(0)
    extent = float((hi - lo).max()) * 1.02 + 1e-9
    centre = (lo + hi) / 2
    R = C.WALL_RASTER
    img = Image.new("L", (R, R), 0)
    draw = ImageDraw.Draw(img)
    for xy in polys:
        p = (xy - centre) / extent * R + R / 2
        draw.polygon([tuple(v) for v in p], fill=255)
    a = np.asarray(img, np.float32) / 255.0
    cover = a.reshape(side, R // side, side, R // side).mean(axis=(1, 3))
    return (cover >= C.WALL_MIN_COVER).astype(np.float32).reshape(-1)


def _build_cache(folders, cache_file, tag):
    start = time.time()
    out = np.zeros((len(folders), C.SIDE * C.SIDE), dtype=np.float32)
    for i, folder in enumerate(folders):
        out[i] = (load_svg_walls(folder / "model.svg") if C.LABELS == "walls"
                  else load_wall_map(folder / "F1_original.png"))
        if (i + 1) % 1000 == 0:
            print(f"  {tag}: {i + 1}/{len(folders)}")
    np.save(cache_file, out)
    print(f"  {tag}: built {out.shape} in {time.time() - start:.0f}s")
    return out


def load_splits(verbose=True):
    """Return (X_train, X_val, X_test) clean wall maps, shape (n, 1024).

    Loads the .npy cache when it exists (shipped with the submission), otherwise
    rebuilds it from the CubiCasa5K dataset using the official split files.
    """
    C.ensure_dirs()
    arrays = []
    dataset_dir = None
    for tag in ("train", "val", "test"):
        suffix = "" if C.LABELS == "ink" else f"_{C.LABELS}"
        cache_file = C.CACHE_DIR / f"cache_{tag}_{C.SIDE}{suffix}.npy"
        if cache_file.exists():
            arrays.append(np.load(cache_file))
            continue
        dataset_dir = dataset_dir or find_dataset_dir()
        if dataset_dir is None:
            raise FileNotFoundError(
                f"{cache_file.name} is missing and the CubiCasa5K dataset was not found "
                f"in any of {[str(p) for p in C.DATASET_LAYOUTS]}. See README.md.")
        arrays.append(_build_cache(read_split(dataset_dir, tag), cache_file, tag))
    X_train, X_val, X_test = arrays
    assert (len(X_train), len(X_val), len(X_test)) == (4200, 400, 400), "unexpected split sizes"
    if verbose:
        print(f"Loaded splits  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}"
              f"  | wall pixels {100 * X_train.mean():.2f}%")
    return X_train, X_val, X_test


def corrupt(clean, rng, dropout=C.DROPOUT, n_blocks=0, block=C.BLOCK):
    """Damage a batch of wall maps: erase n_blocks squares, then drop pixels.

    Training uses pixel dropout only. Block erasure is an out-of-distribution
    test. Damage only ever REMOVES wall pixels, never adds them.
    """
    n, side = clean.shape[0], C.SIDE
    damaged = clean.reshape(n, side, side).copy()
    for i in range(n):
        for _ in range(n_blocks):
            r = int(rng.integers(0, side - block))
            c = int(rng.integers(0, side - block))
            damaged[i, r:r + block, c:c + block] = 0.0
    damaged = damaged.reshape(n, -1)
    if dropout > 0:
        damaged = np.where(rng.random(damaged.shape) < dropout, 0.0, damaged)
    return damaged.astype(np.float32)


def fixed_noisy_sets(X_val, X_test):
    """The fixed damaged validation and test inputs every model is scored on."""
    val_noisy = corrupt(X_val, np.random.default_rng(C.VAL_NOISE_SEED))
    test_noisy = corrupt(X_test, np.random.default_rng(C.TEST_NOISE_SEED))
    return val_noisy, test_noisy
