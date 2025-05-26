import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import librosa
from moviepy.editor import VideoFileClip


def get_statistics(arr, k=4, axis=-1):
    """
    Computes moments up to fourth order of an numpy array along a given axis

    Parameters
    ----------
    arr : np.array
        the numpy array to compute statistics on
    k : int, default = 4
        up to which moment to compute statistics, 4 is the highest implemented
    axis : int, default = -1
        on which axis to compute the statistics
    Returns
    -------
    moments: np.array
       array of moments, shape depends on input shape
    """
    funs = [np.mean, np.std, skew, kurtosis]

    moments = np.stack([funs[i](arr, axis=axis) for i in range(k)])
    return moments


def get_movie_info_moviepy(movie_path):
    """
    Extracts total duration of a movie using moviepy.
    """
    clip = VideoFileClip(movie_path)
    duration = clip.duration
    clip.close()  # Important to release resources
    return duration


def split_movie_into_chunks(movie_path, chunk_interval, chunk_length, seconds_before_chunk):
    """
    Divides a video into fixed-duration chunks with overlapping or non-overlapping options.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    chunk_interval : float, optional
        Interval at which chunks are taken (default is 1.49 seconds).
    chunk_length : float, optional
        Total duration of each chunk in seconds (default is 60 seconds).
    seconds_before_chunk : float, optional
        Length of video included before the chunk's start time (default is 50 seconds).

    Returns
    -------
    list of tuples
        Each tuple represents a chunk with (start_time, end_time).
    """
    video_duration = get_movie_info_moviepy(movie_path)
    chunks = []
    start_time = 0.0

    while start_time < video_duration:
        chunk_start = max(0, start_time - seconds_before_chunk)
        chunk_end = min(chunk_start + chunk_length, video_duration)
        chunks.append((chunk_start, chunk_end))
        start_time += chunk_interval

    return chunks


def extract_audio_features(
        video_path, output_paths,
        chunk_interval,
        chunk_length,
        seconds_before_chunk,
        n_mfcc,
        n_statistics):
    """
       Extracts mono-stereo features - statistical moments of the Mel Frequency Cepstral Coefficients (MFCCs) of the
       sum and the difference of the 2 stereo channels, respectively.

        Parameters
        ----------
        video_path : str
            Path to the video file
        output_paths : iterable of len 2 of strs
            output path for mono and stereo feature, respectively
        chunk_interval : float
            Interval at which chunks are taken (default is 1.49 seconds).
        chunk_length : floa
            Total duration of each chunk in seconds (default is 60 seconds).
        seconds_before_chunk : float
            Length of video included before the chunk's start time (default is 50 seconds).
        n_mfcc: int
            number of MFCCS to compute for each chunk
        n_statistics: int
            number of moments of the MFCCS to compute, highest implemented is 4

        """


    temp_dir = "temp_audio_chunks"  # Define temp_dir
    os.makedirs(temp_dir, exist_ok=True)  # Create temp_dir if it doesn't exist

    # --- HIGHLY RECOMMENDED IMPROVEMENT: Extract entire audio once ---
    print(f"Extracting full audio from {video_path}...")
    full_audio_path = os.path.join(temp_dir, "full_audio.wav")

    # Use moviepy to extract the *entire* audio track
    clip = VideoFileClip(video_path)

    clip.audio.write_audiofile(full_audio_path, verbose=False)

    clip.close()  # Release the video clip resource

    # Load the entire audio with librosa

    y_full, sr = librosa.load(full_audio_path, mono=False)

    print(f"Full audio loaded. Shape: {y_full.shape}, SR: {sr}")
    # --- END IMPROVEMENT ---

    chunks = split_movie_into_chunks(video_path, chunk_interval, chunk_length,
                                     seconds_before_chunk)
    mono_features = []
    stereo_features = []

    print(f"Processing {len(chunks)} audio chunks...")
    for start_time, end_time in tqdm(chunks):  # tqdm for progress bar
        # Calculate sample indices for slicing
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Slice the already loaded audio array
        # Handle cases where chunk might go slightly beyond audio length due to float conversion
        y_chunk = y_full[:, start_sample:end_sample]

        # Ensure we have at least 2 channels for stereo processing
        if y_chunk.shape[0] < 2:
            print(f"Warning: Chunk from {start_time:.2f}s to {end_time:.2f}s is not stereo. Skipping stereo features.")
            # If it's mono, set stereo to zeros or handle as mono-only
            mono = y_chunk[0] if y_chunk.shape[0] == 1 else y_chunk[0]  # Should typically be 1-channel if mono
            stereo = np.zeros_like(mono)  # No stereo difference if mono
        else:
            L = y_chunk[0, :]
            R = y_chunk[1, :]
            mono = (L + R) / np.sqrt(2)
            stereo = (L - R) / np.sqrt(2)

        # Ensure enough samples for MFCCs (e.g., a frame is at least n_fft samples)
        # librosa handles short signals by padding, but extremely short chunks might give weird results.
        # Consider a minimum chunk length or padding strategy if that's an issue.
        if mono.size == 0:
            print(f"Warning: Empty audio chunk from {start_time:.2f}s to {end_time:.2f}s. Skipping.")
            continue

        mono_mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=n_mfcc)
        stereo_mfcc = librosa.feature.mfcc(y=stereo, sr=sr, n_mfcc=n_mfcc)

        mfeatures = get_statistics(mono_mfcc, k=n_statistics).T
        sfeatures = get_statistics(stereo_mfcc, k=n_statistics).T

        mono_features.append(mfeatures.flatten())  # Use .flatten() method
        stereo_features.append(sfeatures.flatten())

    # Format the audio features
    mono_features = np.array(mono_features, dtype="float32")
    stereo_features = np.array(stereo_features, dtype="float32")

    print("Audio feature extraction complete.")
    # Clean up the full_audio.wav file if desired
    os.remove(full_audio_path)
    np.save(output_paths[0], np.array(mono_features))
    np.save(output_paths[1], np.array(stereo_features))


def process_mono_stereo_folder(input_folder, output_folder,
                               chunk_interval, chunk_length,
                               seconds_before_chunk,
                               n_mfcc, n_statistics):
    """
    Processes all video files in a given directory, preserving folder structure, and extracts
     mono-stereo features - statistical moments of the Mel Frequency Cepstral Coefficients (MFCCs) of the
     sum and the difference of the 2 stereo channels, respectively.

    Parameters
    ----------
    input_folder : str
        Path to the parent folder containing videos.
    output_folder : str
        Path to the parent folder where extracted features should be saved.
    chunk_interval : float
        Interval at which chunks are taken (default is 1.49 seconds).
    chunk_length : floa
        Total duration of each chunk in seconds (default is 60 seconds).
    seconds_before_chunk : float
        Length of video included before the chunk's start time (default is 50 seconds).
    n_mfcc: int
        number of MFCCS to compute for each chunk
    n_statistics: int
        number of moments of the MFCCS to compute, highest implemented is 4

    """
    if n_statistics > 4:
        raise NotImplementedError('n_statistics has to be an integer between 1 and 4')
    video_files = []
    if not os.path.exists(input_folder):
        print(f"Error: The path '{input_folder}' does not exist.")
    elif not os.path.isdir(input_folder):
        print(f"Error: The path '{input_folder}' is not a directory.")
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mkv"):
                video_files.append((root, file))
    for root, file in tqdm(video_files, desc="Processing videos"):
        relative_path = os.path.relpath(root, input_folder)
        save_path = os.path.join(output_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path,'mono'), exist_ok=True)
        os.makedirs(os.path.join(save_path,'stereo'), exist_ok=True)

        video_path = os.path.join(root, file)
        output_paths = [os.path.join(save_path,'mono', file.replace(".mkv", ".npy")),
                        os.path.join(save_path,'stereo', file.replace(".mkv", ".npy"))]

        if (not os.path.isfile(output_paths[0]) and not os.path.isfile(output_paths[1])):
            extract_audio_features(video_path, output_paths, chunk_interval, chunk_length, seconds_before_chunk, n_mfcc,
                                   n_statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and extract features")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder where features will be stored")
    parser.add_argument("--chunk_interval", type=float, default=1.49, help="Interval at which chunks are taken")
    parser.add_argument("--chunk_length", type=float, default=4.0, help="Total duration of each chunk in seconds")
    parser.add_argument("--seconds_before_chunk", type=float, default=2.0,
                        help="Length of video included before the chunk's start time")
    parser.add_argument("--n_mfcc", type=int, default=32, help="Number of MFCCs to compute for each chunk")
    parser.add_argument("--n_statistics", type=int, default=4,
                        help="Number of moments   to compute for MFCCS (up to fourth moment is implemented) (i.e. (mean,std,skew,kurtosis))")

    args = parser.parse_args()

    output_folder = f"{args.output_folder}_chunk{args.chunk_interval}_len{args.chunk_length}_before{args.seconds_before_chunk}_nmfcc{args.n_mfcc}_nstats{args.n_statistics}"
    process_mono_stereo_folder(args.input_folder, output_folder, args.chunk_interval, args.chunk_length,
                               args.seconds_before_chunk, args.n_mfcc, args.n_statistics)