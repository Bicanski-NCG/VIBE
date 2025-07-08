import os
import numpy as np
from typing import List, Dict
import zipfile
import numpy as np
from typing import List, Dict

ArrayDict     = Dict[str, np.ndarray]
NestedArrDict = Dict[str, ArrayDict]

def average_nested_dict_arrays(npy_paths: List[str]) -> NestedArrDict:
    """
    Element-wise averages of nested dicts, but inner keys may be missing in
    some files.  Missing keys are *copied through unchanged*.

    Parameters
    ----------
    npy_paths : List[str]
        List of .npy files containing a nested dict structure.

    Returns
    -------
    NestedArrDict
        Same nested structure with averaged arrays.  For keys that appeared in
        only a subset of files the array is identical to the original value(s)
        (no dilution from missing files).
    """
    if not npy_paths:
        raise ValueError("npy_paths list is empty.")

    # ---------- Initialise running sums & counts from first file ------------
    first: NestedArrDict = np.load(npy_paths[0], allow_pickle=True).item()

    sums: NestedArrDict   = {}
    counts: Dict[str, Dict[str, int]] = {}

    for ok, inner in first.items():
        sums[ok]   = {ik: arr.astype(np.float64, copy=True) for ik, arr in inner.items()}
        counts[ok] = {ik: 1 for ik in inner}

    # ---------- Loop over remaining files -----------------------------------
    for path in npy_paths[1:]:
        data: NestedArrDict = np.load(path, allow_pickle=True).item()

        # Outer keys must match exactly
        if data.keys() != sums.keys():
            missing = set(sums) ^ set(data)
            raise KeyError(f"Mismatching outer keys {missing} between files")

        for ok in sums:
            inner_dat = data[ok]
            inner_sum = sums[ok]
            inner_cnt = counts[ok]

            # Visit every key present in *this* file
            for ik, arr in inner_dat.items():
                if ik in inner_sum:
                    if arr.shape != inner_sum[ik].shape:
                        raise ValueError(
                            f"Shape mismatch for ({ok!r}, {ik!r}) in file {path}: "
                            f"expected {inner_sum[ik].shape}, got {arr.shape}"
                        )
                    inner_sum[ik] += arr
                    inner_cnt[ik] += 1
                else:
                    # New key seen for the first time → copy & start count at 1
                    inner_sum[ik] = arr.astype(np.float64, copy=True)
                    inner_cnt[ik] = 1
            # Keys missing in this file are simply ignored (no increment)

    # ---------- Convert sums to means using per-key counts -------------------
    averaged: NestedArrDict = {}
    for ok, inner_sum in sums.items():
        averaged[ok] = {
            ik: inner_sum[ik] / counts[ok][ik]
            for ik in inner_sum
        }

    return averaged

def describe_nested_dict(nested: NestedArrDict) -> None:
    """
    Pretty-print the hierarchy and array shapes of a NestedArrDict.

    Example output
    --------------
    outer_key_1
        inner_key_a : shape=(64, 128), dtype=float64
        inner_key_b : shape=(32, 32),  dtype=float64
    outer_key_2
        inner_key_c : shape=(512,),    dtype=float32
        ...

    Parameters
    ----------
    nested : NestedArrDict
        The dictionary you’d like to inspect.
    """
    for outer_key in sorted(nested):
        print(outer_key)
        inner_dict = nested[outer_key]
        for inner_key in sorted(inner_dict):
            arr = inner_dict[inner_key]
            # Align the colon for nicer columns (optional)
            print(f"    {inner_key:<15}: shape={arr.shape}, dtype={arr.dtype}")


paths = ["many/submissions/submission_ensemble_wot_mononoke_pulpfiction_passepartout_planetearth_39.npy",
         "no-text/submissions/submission_ensemble_chaplin_725.npy"]

save_path_name = "many/submissions/ood"

average = average_nested_dict_arrays(paths)

describe_nested_dict(average)

output_file = f"{save_path_name}.npy"
np.save(output_file, average, allow_pickle=True)

zip_file = f"{save_path_name}.zip"
with zipfile.ZipFile(zip_file, "w") as zipf:
    zipf.write(output_file, os.path.basename(output_file))
print(f"Saved predictions to {zip_file}", flush=True)

