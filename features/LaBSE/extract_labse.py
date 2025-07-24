#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract per-TR Qwen-2.5-7B features (mean of last 4 hidden layers) from TSV transcripts.
Output: one .npy file per TSV, shape (num_tr, 4096)
"""

import argparse
import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel

# ----------------- helpers -------------------------------------------------- #

def convert_seconds(seconds: float) -> str:
    """1 → '0:00:01', 90 → '0:01:30' (only used for metadata)"""
    return str(timedelta(seconds=seconds))

def load_transcript(tsv_path: str) -> List[str]:
    """Return list[str] of transcript lines, one per TR ('' if NaN)."""
    tr_lines = list(pd.read_csv(tsv_path, sep="\t")["text_per_tr"])
    return [" " if str(t) == "nan" else str(t) for t in tr_lines]

# ----------------- feature extraction -------------------------------------- #

@torch.no_grad()
def extract_tsv_features(
    tsv_path: str,
    output_path: str,
    tokenizer,
    model,
    tr: float,
    pre: int = 0,
    post: int = 0,
):
    transcripts = load_transcript(tsv_path)
    num_tr = len(transcripts)

    # For 3 TRs window (2 before, 0 after):
    pre_window = pre
    post_window = post

    window_trs = []
    for idx in range(num_tr):
        start = max(0, idx - pre_window)
        end   = min(num_tr, idx + 1 + post_window)      # exclusive
        window_text = " ".join(transcripts[start:end])
        window_trs.append(window_text)

    for i in range(0, num_tr, 32):
        batch = window_trs[i : i + 32]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in encoded.items()}

        outputs = model(**inputs, output_hidden_states=True)

        attention_mask = inputs["attention_mask"]

        def masked_mean(hidden):
            return (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # ── 1. Low-level phonetic / early-syntax   (layers 3–5) ─────────────────────
        low_layers  = outputs.hidden_states[3:6]              # 3,4,5  (3 layers)
        low_pool    = torch.stack([masked_mean(h) for h in low_layers]).mean(dim=0)   # (B, 768)

        # ── 2. Mid-level lexical-syntax/semantics  (layers 6–8) ─────────────────────
        mid_layers  = outputs.hidden_states[6:9]              # 6,7,8
        mid_pool    = torch.stack([masked_mean(h) for h in mid_layers]).mean(dim=0)   # (B, 768)

        # ── 3. High-level sentence semantics       (layers 8–10)─────────────────────
        high_layers = outputs.hidden_states[8:11]             # 8,9,10
        high_pool   = torch.stack([masked_mean(h) for h in high_layers]).mean(dim=0)  # (B, 768)

        # Determine directory structure
        save_dir = os.path.dirname(output_path)              # out_root/rel_path
        stim_dir = os.path.dirname(save_dir)                # out_root/rel_path/stimulus
        out_root = os.path.dirname(stim_dir)                 # out_root = .../labse_tr1.49
        rel_path = os.path.relpath(save_dir, out_root)       # original relative path

        for feat_name, array in {
            "pool_l3_5":  low_pool.cpu().numpy(),
            "pool_l6_8":  mid_pool.cpu().numpy(),
            "pool_l8_10": high_pool.cpu().numpy(),
        }.items():
            feat_dir = os.path.join(out_root, feat_name, rel_path)
            os.makedirs(feat_dir, exist_ok=True)
            out_file = os.path.join(feat_dir, os.path.basename(output_path))
            if os.path.exists(out_file):
                prev = np.load(out_file)
                array = np.concatenate([prev, array], axis=0)
            np.save(out_file, array)


# ----------------- driver -------------------------------------------------- #

def process_transcripts_folder(
    input_folder: str,
    output_folder: str,
    tokenizer,
    model,
    tr: float,
    pre: int = 0,
    post: int = 0,
):
    transcript_files = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith('.') for part in root.split(os.sep)):
            # skip hidden folders like .ipynb_checkpoints
            continue
        for file in files:
            if file.endswith(".tsv"):
                transcript_files.append((root, file))

    for root, file in tqdm(transcript_files, desc="Extracting features"):
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, rel_path)
        # os.makedirs(save_dir, exist_ok=True)

        tsv_path = os.path.join(root, file)
        out_path = os.path.join(save_dir, file.replace(".tsv", ".npy"))

        if not os.path.isfile(out_path):
            extract_tsv_features(tsv_path, out_path, tokenizer, model, tr, pre, post)


# ----------------- main CLI ------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen-2.5-7B features per TR from TSV transcripts"
    )
    parser.add_argument("--input_folder", required=True,
                        help="Root folder containing .tsv transcripts")
    parser.add_argument("--output_folder", required=True,
                        help="Root folder where .npy features will be stored")
    parser.add_argument("--pre", type=int, default=0,
                        help="Number of TRs before the current one to include (default: 0)")
    parser.add_argument("--post", type=int, default=0,
                        help="Number of TRs after the current one to include (default: 0)")
    parser.add_argument("--tr", type=float, default=1.49,
                        help="Repetition time in seconds (default: 1.49)")
    args = parser.parse_args()

    device = "auto" if torch.cuda.is_available() else "cpu"
    model_name = "setu4993/LaBSE"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True).eval()

    # Output folder name embeds TR for bookkeeping
    out_root = f"{args.output_folder}_pre{args.pre}_post{args.post}_tr{args.tr}"
    process_transcripts_folder(args.input_folder, out_root, tokenizer, model, args.tr, args.pre, args.post)


if __name__ == "__main__":
    main()