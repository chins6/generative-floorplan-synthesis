"""Project paths, random seeds and hyper-parameters shared by every model."""
from pathlib import Path

# ---- paths -------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "cache"            # preprocessed 32x32 arrays (.npy)
MODELS_DIR = PROJECT_DIR / "models"          # best checkpoints (.npz)
RESULTS_DIR = PROJECT_DIR / "results"        # CSV tables
FIGURES_DIR = PROJECT_DIR / "figures"        # PNG figures for report / slides

# The Kaggle download unzipped as archive/cubicasa5k/cubicasa5k, or the older
# dataset/cubicasa5k layout. Only needed if the cache has to be rebuilt.
DATASET_LAYOUTS = [
    PROJECT_DIR / "archive" / "cubicasa5k" / "cubicasa5k",
    PROJECT_DIR / "dataset" / "cubicasa5k",
]

# ---- data --------------------------------------------------------------------
SIDE = 32                 # wall maps are SIDE x SIDE
WALL_THRESHOLD = 200      # grey level below which a source pixel is "ink"
DROPOUT = 0.80            # training corruption: fraction of pixels erased
BLOCK = 8                 # side of an erased square (out-of-distribution test)

# Fixed corruption seeds, so every model is scored on the SAME damaged images
VAL_NOISE_SEED = 7
TEST_NOISE_SEED = 99

# ---- training (shared by all three models, so the comparison is fair) ---------
WALL_WEIGHT = 6.0         # class weight on wall pixels in the BCE loss
LEARNING_RATE = 0.003
BATCH_SIZE = 32
EPOCHS = 90
TRAIN_SEED = 42

# ---- model-specific defaults ---------------------------------------------------
DAE_WIDTH = 16            # filters in the first / last conv stage
VAE_WIDTH = 16
VAE_BETA = 1.0            # weight of the KL term (1.0 = the true ELBO)
VAE_KL_WARMUP = 10        # epochs over which beta is ramped up from 0
GNN_HIDDEN = 64           # node embedding size
GNN_LAYERS = 4            # GraphSAGE message-passing layers
GNN_PATCH = 4             # each graph node is a PATCH x PATCH block of pixels


# ---- label set ---------------------------------------------------------------
# "ink"   : every dark pixel of F1_original.png (the mid-review representation)
# "walls" : only the annotated Wall polygons of model.svg (CubiCasa5K's own labels)
LABELS = "ink"
WALL_RASTER = 512         # walls are drawn at 512x512 before reduction to 32x32
WALL_MIN_COVER = 0.25     # a 32x32 cell is wall if >= 25% of its area is wall
                          # (0.05 gives 2-cell-thick walls, 0.35 breaks thin walls apart)


def set_labels(name):
    """Switch label set. 'walls' keeps its cache, models, results and figures in
    separate files / sub-folders, so the two experiments never overwrite each other."""
    global LABELS, MODELS_DIR, RESULTS_DIR, FIGURES_DIR
    assert name in ("ink", "walls"), name
    LABELS = name
    sub = "" if name == "ink" else name
    MODELS_DIR = PROJECT_DIR / "models" / sub
    RESULTS_DIR = PROJECT_DIR / "results" / sub
    FIGURES_DIR = PROJECT_DIR / "figures" / sub


def ensure_dirs():
    for folder in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
