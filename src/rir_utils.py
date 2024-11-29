import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import resample
from scipy.io import wavfile
import librosa
def load_rir(filepath):
    """
    Load Room Impulse Response (RIR) from a file.
    
    Parameters:
    - filepath: str, path to the RIR file (e.g., .wav).
    
    Returns:
    - rir: numpy array, the RIR signal.
    - sampling_rate: int, the sampling rate of the file.
    """
    sampling_rate, rir = wavfile.read(filepath)
    # Normalize the RIR
    rir = rir / np.max(np.abs(rir))
    return rir, sampling_rate

def compute_edc(rir):
    """
    Compute the Energy Decay Curve (EDC) from the Room Impulse Response (RIR).
    """
    energy = rir**2
    edc = np.cumsum(energy[::-1])[::-1]  # Reverse cumulative sum
    edc = edc / edc[0]  # Normalize to 1
    return edc

def resample_rir(rir, original_rate, target_rate):
    """
    Resample the RIR signal to a new sampling rate.
    """
    
    print(f"Original Sample Rate: {original_rate} Hz")
    
    # Resample the audio to the target sample rate
    resampled_data = librosa.resample(rir, orig_sr=original_rate, target_sr=target_rate)
    print(f"Resampling to {target_rate} Hz...")
    return resampled_data

def compute_t60(rir, sampling_rate=8000, resample=False, original_rate=8000, target_rate=8000, decay_range=(-5, -35)):
    """
    Calculate the T60 reverberation time from a Room Impulse Response (RIR).
    
    Parameters:
    - rir: array-like, the room impulse response signal.
    - sampling_rate: int, the sampling rate of the RIR in Hz.
    - resample: bool, whether to resample the RIR to a new sampling rate.
    - original_rate: int, the original sampling rate in Hz.
    - target_rate: int, the target sampling rate in Hz (used if resample is True).
    - decay_range: tuple, the dB range (start, end) for linear regression.
    
    Returns:
    - T60: float, the estimated reverberation time in seconds.
    """
    if resample and original_rate != target_rate:
        rir = resample_rir(rir, original_rate, target_rate)
        sampling_rate = target_rate

    # Compute EDC
    edc = compute_edc(rir)
    
    # Convert EDC to decibel scale
    edc_db = 10 * np.log10(edc + 1e-10)  # Add small value to avoid log(0)

    # Find indices corresponding to decay range
    start_idx = np.where(edc_db <= decay_range[0])[0][0]
    end_idx = np.where(edc_db <= decay_range[1])[0][0]

    # Extract time and EDC values for fitting
    time = np.linspace(0, len(rir) / sampling_rate, len(edc_db))
    time_fit = time[start_idx:end_idx]
    edc_db_fit = edc_db[start_idx:end_idx]

    # Perform linear regression on the decay range
    slope, intercept, _, _, _ = stats.linregress(time_fit, edc_db_fit)

    # Calculate T60
    T60 = -60 / slope
    return T60, time, edc_db, time_fit, edc_db_fit, slope, intercept

def plot_results(time, edc_db, time_fit, edc_db_fit, slope, intercept, T60):
    """
    Plot the EDC and the linear fit for the selected range, marking the T60 time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, edc_db, label="EDC (dB)")
    plt.plot(time_fit, edc_db_fit, 'g', label="Selected Range (-5 to -35 dB)")
    plt.plot(time_fit, slope * time_fit + intercept, 'r--', label="Linear Fit")
    
    # Mark T60 on the plot
    plt.axvline(T60, color='b', linestyle='--', label=f"T60 = {T60:.2f} s")
    plt.text(T60, -30, f"T60 = {T60:.2f} s", color='b', fontsize=10, va='bottom', ha='left')

    plt.xlabel("Time (s)")
    plt.ylabel("Energy Decay (dB)")
    plt.title(f"Reverberation Time (T60): {T60:.2f} seconds")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Filepath to the RIR file
    filepath = "C:\\Users\\arifr\\SS_Dataset_preparation\\MIT_RIR\\h001_Bedroom_65txts.wav"  # Replace with the actual path to your RIR file

    # Load RIR
    rir, original_rate = load_rir(filepath)
    target_rate = 8000    # Target sampling rate
    resample_bool = True  # Whether to resample the RIR
    plot = True # Whether to plot results
    # Compute T60
    T60, time, edc_db, time_fit, edc_db_fit, slope, intercept = compute_t60(
        rir, 
        resample=resample_bool, 
        original_rate=original_rate, 
        target_rate=target_rate
    )

    # Print and plot results
    print(f"Estimated T60: {T60:.2f} seconds")
    if plot:
         plot_results(time, edc_db, time_fit, edc_db_fit, slope, intercept, T60)
