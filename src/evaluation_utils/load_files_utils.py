import json
import os

import librosa


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""

    pass


class ErrorLoadingMetadataError(Exception):
    """Custom exception for Loading Metadataerrors."""

    pass


def load_and_validate_config(file_path):
    """
    Loads and validates a JSON config file.

    Args:
        file_path (str): Path to the config.json file.

    Returns:
        dict: Parsed JSON object if valid.
        str: Error message if validation fails.
    """
    # Check if the file path exists
    if not os.path.exists(file_path):
        print("Error: Config file does not exist.", flush=True)
        raise ConfigValidationError

    # Load the JSON file
    try:
        with open(file_path, "r") as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        print("Error: Invalid JSON format. Details: {e}", flush=True)
        raise ConfigValidationError

    # Validate the presence of the 1st hierarchy keys
    esseential_keys = ["target_sample_rate", "directories"]
    for key in esseential_keys:
        if key not in config_data:
            print(
                f"Error: Missing key '{key}' the config file. Invalid input config",
                flush=True,
            )
            raise ConfigValidationError

    # Validate that "directories" is a dictionary
    directories = config_data["directories"]
    if not isinstance(directories, dict):
        print("Error: 'directories' must be a dictionary.", flush=True)
        raise ConfigValidationError

    # Validate required keys within "directories"
    required_keys = ["dataset_directory", "separatd_audios_directory"]
    for key in required_keys:
        if key not in directories:
            print(f"Error: Missing key '{key}' in 'directories'.", flush=True)
            raise ConfigValidationError

    return config_data  # Return the valid config data


def load_metadata(dataset_dir: str):
    """
    Load the metadata file using a path from the environment variable.
    Returns:
        List of metadata entries.
    Raises:
        FileNotFoundError: If the file path is invalid.
    """
    try:
        metadata_file_path = os.path.join(dataset_dir, "metadata.json")
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file '{metadata_file_path}' not found.")
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        raise ErrorLoadingMetadataError


def load_audio_and_resmaple(audio_path: str, target_sample_rate: int):
    resampled = False
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        resampled = True
    return audio, resampled
