import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd

from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer

def convert_seconds(seconds):
    return str(timedelta(seconds=seconds))

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_after_substring(s, substring):
    index = s.find(substring)
    if index != -1:
        return s[index + len(substring):]
    else:
        return None  # or raise an error or return the original string
    
movie_dict = {
    "movie10_bourne":  "The Bourne supremacy. Duration ~100 minutes. Total Parts: 10",
    "movie10_figures": "Hidden figures. Duration ~120 minutes. Total Parts: 12",
    "movie10_life": "Life Disc one of four: “Challenges of life, reptiles and amphibian mammals”. DVD set was narrated by David Attenborough. Duration, and lasted ~50 minutes. Total Parts: 5",
    "movie10_wolf":  "The wolf of wall street. Duration ~170 minutes. Total Parts 16"
}

def extract_tsv_features(tsv_path, output_path, tokenizer, model, tr, chunk_length, seconds_before_chunk):
    transcripts = list(pd.read_csv(tsv_path, sep="\t")["text_per_tr"])
    transcripts = [" " if str(t)=="nan" else t for t in transcripts]
    timestamps = [convert_seconds(int(i*tr)) for i in range(len(transcripts))]
    num_tr_before = seconds_before_chunk // tr
    num_tr_after = (chunk_length - seconds_before_chunk) // tr

    metadata = "=== METADATA ===\n"
    if "friends" in tsv_path:
        metadata = f"{metadata} TV Series: {tsv_path.split("/")[-1][:-5]} Part {tsv_path.split("/")[-1][-5]} out of two parts: a and b"
    else:
        metadata = f"{metadata} Movie: {movie_dict[tsv_path.split("/")[-1][:-6]]}. Current part {tsv_path.split("/")[-1][-5]}"


    formatted_transcripts = [
        f"[Time: {timestamps[i]}s] {transcripts[i]}" for i in range(len(transcripts))
    ]
    
    features = []

    for i in range(len(timestamps)):
        start_idx = int(max(0, i-num_tr_before))
        end_idx = int(min(len(timestamps), i+ num_tr_after+1))
        dialogue_interest = formatted_transcripts[i]
        context_text = f"[Metadata]: {metadata}\n[Instructions]: Extract the meaning of TARGET TEXT======== \n[Context]:\n" + "\n".join(formatted_transcripts[start_idx:end_idx])
        target_phrase = f"TARGET TEXT TO FOCUS: {dialogue_interest}"
        context_text = f"{context_text}\n{target_phrase}"
        
        encoded = tokenizer(
            context_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=True
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"][0]

        target_start_char = context_text.find(target_phrase)

        focus_token_indices = [
            i for i, (start, end) in enumerate(offsets)
            if start < target_start_char + len(target_phrase) and end > target_start_char
        ]

        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of layers
            last_hidden = hidden_states[-1]  # Shape: [1, seq_len, hidden_dim]

        target_embeddings = last_hidden[0, focus_token_indices, :]  # [num_tokens, hidden_dim]
        sentence_embedding = target_embeddings.mean(dim=0, keepdim=True)[0]  # [1, hidden_dim]

        features.append(sentence_embedding.detach().cpu().numpy())

    np.save(output_path, np.array(features))

def process_transcripts_folder(input_folder, output_folder, tokenizer, model, tr, chunk_length, seconds_before_chunk):
    """
    Processes all video files in a given directory, preserving folder structure, and extracts features.

    Parameters
    ----------
    input_folder : str
        Path to the parent folder containing videos.
    output_folder : str
        Path to the parent folder where extracted features should be saved.
    config : dict
        Configuration dictionary for feature extraction.
    intern_model : model
        The feature extraction model.
    """
    transcript_files = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue
        for file in files:
            if file.endswith(".tsv"):
                transcript_files.append((root, file))
    
    for root, file in tqdm(transcript_files, desc="Processing Transcripts"):
        relative_path = os.path.relpath(root, input_folder)
        save_path = os.path.join(output_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        
        tsv_path = os.path.join(root, file)
        output_path = os.path.join(save_path, file.replace(".tsv", ".npy"))
        if not os.path.isfile(output_path):
            extract_tsv_features(tsv_path, output_path, tokenizer, model, tr, chunk_length, seconds_before_chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and extract features")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where features will be stored")
    parser.add_argument("--tr", type=float, default=1.49, help="Interval at which chunks are taken")
    parser.add_argument("--chunk_length", type=float, default=60, help="Total duration of each chunk in seconds")
    parser.add_argument("--seconds_before_chunk", type=float, default=50, help="Length of video included before the chunk's start time")
    
    args = parser.parse_args()
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    
    output_folder = f"{args.output_folder}_tr{args.tr}_len{args.chunk_length}_before{args.seconds_before_chunk}"
    process_transcripts_folder(args.input_folder, output_folder, tokenizer, model, args.tr, args.chunk_length, args.seconds_before_chunk)
