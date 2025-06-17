

"""
Compare legacy loader (tests.ref.ref_loader) and new loader
(algonauts.data.loader) on an in‑memory synthetic dataset.

Run:
    pytest -q tests/test_loader.py
"""

import os, sys, types, shutil, h5py, numpy as np, torch, pytest
from pathlib import Path
import importlib

# ---------------------------------------------------------------------
# Stub external deps (wandb, logger) before importing new loader
# ---------------------------------------------------------------------
wandb_stub = types.ModuleType("wandb")
wandb_stub.log = lambda *_, **__: None
sys.modules["wandb"] = wandb_stub

logger_stub = types.ModuleType("algonauts.utils.logger")
logger_stub.info = lambda *_, **__: None
sys.modules["algonauts.utils.logger"] = logger_stub

# now safe to import loaders
old_loader = importlib.import_module("ref.ref_loader")
new_loader = importlib.import_module("algonauts.data.loader")
old_data = importlib.import_module("ref.ref_data")  # legacy dataset impl
new_data = importlib.import_module("algonauts.data.data")  # new dataset impl

FMRI_Dataset_old = old_data.FMRI_Dataset
FMRI_Dataset_new = new_data.FMRI_Dataset

# ---------------------------------------------------------------------
# Dummy on‑disk dataset shared by both loaders
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("algonauts_loader")
    # fmri
    func_dir = root / "sub-01" / "func"
    func_dir.mkdir(parents=True)
    with h5py.File(func_dir / "clip.h5", "w") as h5:
        h5.create_dataset("ses-01_task-clip1", data=np.random.randn(10, 5).astype("float32"))
    (root / "sub-01" / "atlas").mkdir(parents=True)
    (root / "sub-01/atlas/atlas.nii.gz").touch()
    # features
    (root / "features_audio").mkdir()
    (root / "features_video").mkdir()
    np.save(root / "features_audio/clip1.npy", np.random.randn(10, 16).astype("float32"))
    np.save(root / "features_video/clip1.npy", np.random.randn(10, 32).astype("float32"))
    # dummy normalization_stats.pt
    torch.save({}, root / "normalization_stats.pt")
    yield root
    shutil.rmtree(root)

# ---------------------------------------------------------------------
# Minimal config object for both loaders
# ---------------------------------------------------------------------
class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

@pytest.fixture(scope="session")
def common_cfg(dummy_root):
    cfg = Cfg(
        # paths / dims
        data_dir=dummy_root,
        features_dir=str(dummy_root),
        features={"audio": "features_audio", "video": "features_video"},
        input_dims={"audio": 16, "video": 32},
        modalities=["audio", "video"],
        # data split
        val_name="one",
        val_run="all",
        filter_name=None,
        # loader params
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
        # augmentation / norm
        train_noise_std=0.0,
        use_normalization=False,
        oversample_factor=1,
        stratification_variable=False,
        normalize_validation_bold=False,
    )
    # ensure torch.load finds stats
    os.chdir(dummy_root)
    return cfg

# ---------------------------------------------------------------------
# Helper equality checks
# ---------------------------------------------------------------------
def tensors_close(a, b, atol=1e-6):
    if torch.is_tensor(a):
        return torch.allclose(a, b, atol=atol)
    return np.allclose(a, b, atol=atol)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_loader_lengths_and_first_batch(common_cfg):
    # legacy
    torch.manual_seed(0)
    train_old, val_old, full_old = old_loader.get_data_loaders(common_cfg)

    # new
    torch.manual_seed(0)
    train_new, val_new = new_loader.get_train_val_loaders(common_cfg)
    full_new = new_loader.get_full_loader(common_cfg)

    # --- dataset lengths identical ---
    assert len(train_old.dataset) == len(train_new.dataset)
    assert len(val_old.dataset) == len(val_new.dataset)
    assert len(full_old.dataset) == len(full_new.dataset)

    # --- first batch identical ---
    bo = next(iter(train_old))
    bn = next(iter(train_new))
    assert bo.keys() == bn.keys()
    for k in bo:
        assert tensors_close(bo[k], bn[k])

def test_consistency_of_collate(common_cfg):
    ds_old = FMRI_Dataset_old(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities,
        noise_std=0.0)
    ds_new = FMRI_Dataset_new(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities,
        noise_std=0.0)

    b_old = [ds_old[0], ds_old[0]]
    b_new = [ds_new[0], ds_new[0]]
    batch_old = old_loader.collate_fn(b_old)
    batch_new = new_loader.collate_fn(b_new)
    for k in batch_old:
        assert tensors_close(batch_old[k], batch_new[k])

def test_split_fn_identical(common_cfg):
    ds_old = FMRI_Dataset_old(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities)
    ds_new = FMRI_Dataset_new(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities)
    tr_o, val_o = old_loader.split_dataset_by_name(ds_old, val_name="clip")
    tr_n, val_n = new_loader.split_dataset_by_name(ds_new, val_name="clip")
    assert len(tr_o) == len(tr_n) == 0
    assert len(val_o) == len(val_n) == 1

def test_group_weights_identical(common_cfg):
    ds_old = FMRI_Dataset_old(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities)
    ds_new = FMRI_Dataset_new(
        common_cfg.data_dir,
        feature_paths={m: str(Path(common_cfg.features_dir)/p)
                       for m, p in common_cfg.features.items()},
        input_dims=common_cfg.input_dims,
        modalities=common_cfg.modalities)
    w_old = old_loader.make_group_weights(ds_old, filter_on="subject_id")
    w_new = new_loader.make_group_weights(ds_new, filter_on="subject_id")
    assert torch.allclose(w_old, w_new)