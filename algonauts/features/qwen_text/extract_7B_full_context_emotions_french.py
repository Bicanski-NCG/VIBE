#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract per-TR Qwen-2.5-7B features (mean of last 4 hidden layers) from TSV transcripts.
Output: one .npy file per TSV, shape (num_tr, 4096)
"""

import argparse
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------- helpers -------------------------------------------------- #

def convert_seconds(seconds: float) -> str:
    """1 → '0:00:01', 90 → '0:01:30' (only used for metadata)"""
    return str(timedelta(seconds=seconds))

def load_transcript(tsv_path: str) -> List[str]:
    """Return list[str] of transcript lines, one per TR ('' if NaN)."""
    tr_lines = list(pd.read_csv(tsv_path, sep="\t")["text_per_tr"])
    return [" " if str(t) == "nan" else str(t) for t in tr_lines]

# ----------------- feature extraction -------------------------------------- #

def build_prompt(
    transcripts: List[str],
    metadata: str
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Concatenate metadata + newline-separated transcript lines.

    Returns
    -------
    full_text : str
    char_spans : list of (start, end) for each TR line *in the full_text*
                 (metadata is excluded – its tokens will be ignored later)
    """
    parts = [f"[Metadata]: {metadata}\n"]
    char_spans = []
    pos = len(parts[0])

    for line in transcripts:
        start = pos
        parts.append(line + "\n")
        pos += len(line) + 1   # +1 for '\n'
        char_spans.append((start, pos))

    return "".join(parts), char_spans


def tokens_by_tr(offsets, char_spans):
    """
    Map each token index to a TR index, based on character offsets.

    Parameters
    ----------
    offsets : Tensor (seq_len, 2) – tokenizer char spans
    char_spans : list[(start, end)] – TR char spans

    Returns
    -------
    list[list[int]] – token indices per TR
    """
    per_tr = [[] for _ in char_spans]

    for tok_idx, (tok_start, tok_end) in enumerate(offsets.tolist()):
        # skip metadata tokens (those before first TR start)
        if tok_end <= char_spans[0][0]:
            continue
        # assign token to the first TR whose span it overlaps
        for tr_idx, (start, end) in enumerate(char_spans):
            if tok_start < end and tok_end > start:
                per_tr[tr_idx].append(tok_idx)
                break
    return per_tr


@torch.no_grad()
def extract_tsv_features(
    tsv_path: str,
    output_path: str,
    tokenizer,
    model,
    tr: float,
):
    transcripts = load_transcript(tsv_path)
    num_tr = len(transcripts)

    # ------------- build metadata ----------------------------------------- #
    fname = os.path.basename(tsv_path)
    if "friends" in tsv_path.lower():
        # .../friends_s02e05_partA.tsv  →  'TV Series: friends_s02e05 Part A of two parts'
        metadata = (
            f"Série télévisée : {fname[:-4]} "
            f"Partie {fname[-5]} de deux parties : a et b"
        )
    elif 'movie10' in tsv_path.lower():
        movie_dict = {
            "movie10_bourne":  "The Bourne Supremacy (~100 min, 10 parts)",
            "movie10_figures": "Hidden Figures (~120 min, 12 parts)",
            "movie10_life":    "Life (Attenborough, disc 1, ~50 min, 5 parts)",
            "movie10_wolf":    "The Wolf of Wall Street (~170 min, 16 parts)",
        }
        key = fname[:-6]               # strip '_?.tsv'
        part_no = fname[-5]
        metadata = (
            f"Film: {movie_dict.get(key, key)}. "
            f"Partie actuelle :  {part_no}"
        )
    elif 'ood' in tsv_path.lower():

        movie_dict = {'ood_mononoke': "Princess Mononoke (~133 min, 13 parts)",
                      'ood_planetearth': "Planet Earth (Season II, Episode I, ""Mountains"") (~48 mins, 4 parts)",
                      'ood_pulpfiction': "Pulp Fiction: (~153 min, 15 parts)",
                      'ood_wot': "World of Tomorrow by Don Herzfeldt (Animated Short Film, ~16 min, 2 parts)"}
        
        key = fname[:-6]               # strip '_?.tsv'
        part_no = fname[-5]
        if "passepartout" not in key:
            metadata = (
                f"Film: {movie_dict.get(key, key)}. "
                f"Partie actuelle : {part_no}")
        else:

            if part_no =="1":
                episode = "Saison 2, Épisode 7"
            else:
                episode = "Saison 2, Épisode 8"

            
            metadata = (
                f"Série télévisée : Passe-Partout (1977-1993) "
                f"Épisode:  {episode}")


    
    emotional_priming_phrase = (
        "[Instruction] : expert·e en empathie du public, examinez minutieusement l’état émotionnel "
    "des personnages ainsi que l’atmosphère émotionnelle globale de la scène suivante. "
    "Considérez et encodez les émotions prédominantes que ce moment est conçu pour susciter "
    "(p. ex. joie, tristesse, tension, surprise, peur, colère, soulagement, ennui, émerveillement, etc.). "
    "Soyez attentif·ve aux transitions émotionnelles et à leur intensité. "
    )
    # ------------------------------------------------------------------

    # Combine the priming phrase with the existing metadata
    metadata = f"{emotional_priming_phrase}[Metadata]: {metadata}"
    # ------------- encode full transcript -------------------------------- #
    full_text, char_spans = build_prompt(transcripts, metadata)

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    input_ids = encoded["input_ids"].to(model.device)           # (1, seq_len)
    offsets = encoded["offset_mapping"][0]                      # (seq_len, 2)

    # ------------- forward pass ------------------------------------------ #
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states: tuple(len = n_layers+1) of (1, seq_len, hidden_dim)
    hidden_last4 = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)[0]   # (seq_len, 4096)

    # ------------- pool by TR -------------------------------------------- #
    per_tr_tokens = tokens_by_tr(offsets, char_spans)           # list[list[int]]
    features = np.zeros((num_tr, hidden_last4.size(1)), dtype=np.float32)

    for i, token_idxs in enumerate(per_tr_tokens):
        if token_idxs:  # non-empty
            features[i] = hidden_last4[token_idxs].mean(dim=0).float().cpu().numpy()
        # else leave zeros (silent TR)

    np.save(output_path, features)


# ----------------- driver -------------------------------------------------- #

def process_transcripts_folder(
    input_folder: str,
    output_folder: str,
    tokenizer,
    model,
    tr: float
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
        os.makedirs(save_dir, exist_ok=True)

        tsv_path = os.path.join(root, file)
        out_path = os.path.join(save_dir, file.replace(".tsv", ".npy"))

        if not os.path.isfile(out_path):
            extract_tsv_features(tsv_path, out_path, tokenizer, model, tr)


# ----------------- main CLI ------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen-2.5-7B features per TR from TSV transcripts"
    )
    parser.add_argument("--input_folder", default="/u/janik/algonauts/MultimodalBrainModel/algonauts_2025.competitors/stimuli/transcripts/",
                        help="Root folder containing .tsv transcripts")
    parser.add_argument("--output_folder", default="/u/shdixit/MultimodalBrainModel/Features/Text/Qwen2.5_14_Full_French",
                        help="Root folder where .npy features will be stored")
    parser.add_argument("--tr", type=float, default=1.49,
                        help="Repetition time in seconds (default: 1.49)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-14B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
        .to(device)
        .eval()
    )

    # Output folder name embeds TR for bookkeeping
    out_root = f"{args.output_folder}_tr{args.tr}"
    process_transcripts_folder(args.input_folder, out_root, tokenizer, model, args.tr)


if __name__ == "__main__":
    main()