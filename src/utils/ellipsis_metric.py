import librosa
import numpy as np

# -----------------------------------------------------------------------------
# Step 1: Load the Audio Signals
# -----------------------------------------------------------------------------
# Load three audio signals:
#   - s1: the first source signal.
#   - s2: the second source signal.
#   - m : the observed mixture signal.
# We set sr=None so that the audio is loaded at its original sampling rate.
s1, sr1 = librosa.load(
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_output_prev\\0a2e3b35-345a-496d-9cb8-9af73911410a_spk1_corrected.wav",
    sr=None,
)
s2, sr2 = librosa.load(
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_output_prev\\0a2e3b35-345a-496d-9cb8-9af73911410a_spk2_corrected.wav",
    sr=None,
)
m, sr_m = librosa.load(
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_input_prev\\0a2e3b35-345a-496d-9cb8-9af73911410a.wav",
    sr=None,
)

# -----------------------------------------------------------------------------
# Step 2: Check Sampling Rates
# -----------------------------------------------------------------------------
# Ensure that all three signals have the same sampling rate.
if sr1 != sr2 or sr1 != sr_m:
    raise ValueError("Sampling rates of the audio signals do not match!")

# -----------------------------------------------------------------------------
# Step 3: Print Signal Lengths and Determine Trimming
# -----------------------------------------------------------------------------
# Print original lengths in samples and in seconds.
print("Original Lengths:")
print("  s1: {} samples ({:.2f} seconds)".format(len(s1), len(s1) / sr1))
print("  s2: {} samples ({:.2f} seconds)".format(len(s2), len(s2) / sr2))
print("  m : {} samples ({:.2f} seconds)".format(len(m), len(m) / sr_m))

# Compute the minimum length among the three signals.
min_length = min(len(s1), len(s2), len(m))
print(
    "\nMinimum length among the signals: {} samples ({:.2f} seconds)".format(
        min_length, min_length / sr1
    )
)

# Calculate how many samples each signal will be trimmed by.
trim_s1 = len(s1) - min_length
trim_s2 = len(s2) - min_length
trim_m = len(m) - min_length

print("\nTrimming Information:")
print("  s1 will be trimmed by {} samples".format(trim_s1))
print("  s2 will be trimmed by {} samples".format(trim_s2))
print("  m  will be trimmed by {} samples".format(trim_m))

# Determine which audio is the longest.
if len(s1) >= len(s2) and len(s1) >= len(m):
    print(
        "\ns1 is the longest audio by {} samples compared to the minimum.".format(
            trim_s1
        )
    )
elif len(s2) >= len(s1) and len(s2) >= len(m):
    print(
        "\ns2 is the longest audio by {} samples compared to the minimum.".format(
            trim_s2
        )
    )
elif len(m) >= len(s1) and len(m) >= len(s2):
    print(
        "\nThe mixture m is the longest audio by {} samples compared to the minimum.".format(
            trim_m
        )
    )
else:
    print("\nAt least two signals share the maximum length.")

# -----------------------------------------------------------------------------
# Step 4: Trim Signals to the Same Length
# -----------------------------------------------------------------------------
# Trim all signals to the minimum length.
s1 = s1[:min_length]
s2 = s2[:min_length]
m = m[:min_length]

# -----------------------------------------------------------------------------
# Step 5: Prepare the Data Matrix and Solve the Least-Squares Problem
# -----------------------------------------------------------------------------
# Let N be the number of samples (after trimming).
N = len(m)

# Create matrix S with shape (N, 2) where the first column is s1 and the second is s2.
S = np.column_stack((s1, s2))

# We want to solve for coefficients w = [a, b]^T in the model:
#    m_hat = S w
# that minimizes the squared error:
#    J(w) = || m - S w ||^2
#
# Expanding J(w):
#    J(w) = m^T m - 2 w^T S^T m + w^T S^T S w
#
# To minimize J(w), we differentiate with respect to w and set the derivative to zero,
# which leads to the normal equation:
#    S^T S w = S^T m
#
# The solution is:
#    w = (S^T S)^{-1} S^T m

w, residuals, rank, s_vals = np.linalg.lstsq(S, m, rcond=None)
a, b = w  # Optimal coefficients

# -----------------------------------------------------------------------------
# Step 6: Reconstruct the Mixture and Evaluate the Reconstruction
# -----------------------------------------------------------------------------
# Reconstruct the mixture using the optimal coefficients:
m_hat = a * s1 + b * s2

# Compute the Mean Squared Error (MSE) between the original mixture and the reconstructed mixture.
mse = np.mean((m - m_hat) ** 2)

# -----------------------------------------------------------------------------
# Step 7: Print the Results
# -----------------------------------------------------------------------------
print("\nOptimal coefficients:")
print("  a =", a)
print("  b =", b)
print("Mean Squared Error of the reconstruction:", mse)
