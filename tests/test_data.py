"""
Compare legacy (tests.ref.ref_data) and current (algonauts.data.data)
implementations on an in-memory dummy dataset.

Run:
    pytest -q tests/test_data.py
"""
import importlib, os, h5py, numpy as np, torch, shutil, pytest
from pathlib import Path
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# dynamic imports
# ------------------------------------------------------------------
old = importlib.import_module("ref.ref_data")
new = importlib.import_module("algonauts.data.data")

# ------------------------------------------------------------------
# dummy dataset on disk
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("algonauts_dummy")
    # 10 × 5 fmri
    func_dir = root / "sub-01" / "func"
    func_dir.mkdir(parents=True)
    # atlas placeholder
    atl = root / "sub-01" / "atlas"
    atl.mkdir(parents=True)
    (atl / "atlas.nii.gz").touch()
    for name in ["one", "two", "three"]:
        h5name = f"ses-01_task-{name}1"
        with h5py.File(func_dir / f"{name}1.h5", "w") as h5:
            h5.create_dataset(h5name, data=np.random.randn(10, 5).astype("float32"))
        # features
        for m, d, D in [("audio", 16, "features_audio"),
                        ("video", 32, "features_video")]:
            p = root / D
            p.mkdir(exist_ok=True)
            np.save(p / f"{name}1.npy", np.random.randn(10, d).astype("float32"))
    yield root
    shutil.rmtree(root)

# ------------------------------------------------------------------
# dataset fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def ds_old(dummy_root):
    return old.FMRI_Dataset(
        root_folder_fmri=dummy_root,
        feature_paths={"audio": dummy_root/"features_audio",
                       "video": dummy_root/"features_video"},
        input_dims={"audio": 16, "video": 32},
        modalities=["audio", "video"])

@pytest.fixture(scope="session")
def ds_new(dummy_root):
    return new.FMRI_Dataset(
        root_folder_fmri=dummy_root,
        feature_paths={"audio": dummy_root/"features_audio",
                       "video": dummy_root/"features_video"},
        input_dims={"audio": 16, "video": 32},
        modalities=["audio", "video"])

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def tensors_equal(a, b, atol=1e-6):
    if torch.is_tensor(a):
        return torch.allclose(a, b, atol=atol)
    return np.allclose(a, b, atol=atol)

# ------------------------------------------------------------------
# tests
# ------------------------------------------------------------------
def test_len(ds_old, ds_new):
    assert len(ds_old) == len(ds_new) == 3

@pytest.mark.parametrize("idx", [0])
def test_getitem(ds_old, ds_new, idx):
    s_old = ds_old[idx]
    s_new = ds_new[idx]
    for a, b in zip(s_old, s_new):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                assert tensors_equal(a[k], b[k])
        else:
            assert tensors_equal(a, b)

def test_batch_equal(ds_old, ds_new):
    g = torch.Generator().manual_seed(42)
    dl_old = DataLoader(ds_old, batch_size=2, shuffle=True, generator=g,
                        collate_fn=old.collate_fn, num_workers=0)
    g = torch.Generator().manual_seed(42)
    dl_new = DataLoader(ds_new, batch_size=2, shuffle=True, generator=g,
                        collate_fn=new.collate_fn, num_workers=0)
    
    for batch_old, batch_new in zip(dl_old, dl_new):
        assert batch_old.keys() == batch_new.keys()
        for k in batch_old:
            assert tensors_equal(batch_old[k], batch_new[k])

def test_collate_fn_equal(ds_old, ds_new):
    batch_old = [ds_old[0], ds_old[0]]
    batch_new = [ds_new[0], ds_new[0]]
    c_old = old.collate_fn(batch_old)
    c_new = new.collate_fn(batch_new)
    assert c_old.keys() == c_new.keys()
    for k in c_old:
        assert tensors_equal(c_old[k], c_new[k])

def test_split_dataset_equal(ds_old, ds_new):
    tr_o, val_o = old.split_dataset_by_name(ds_old, val_name="one")
    tr_n, val_n = new.split_dataset_by_name(ds_new, val_name="one")
    assert len(tr_o) == len(tr_n) == 2
    assert len(val_o) == len(val_n) == 1

def test_group_weights_equal(ds_old, ds_new):
    w_old = old.make_group_weights(ds_old, "subject_id")
    w_new = new.make_group_weights(ds_new, "subject_id")
    assert torch.allclose(w_old, w_new)