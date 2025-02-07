import librosa
import numpy as np

# -----------------------------------------------------------------------------
# Step 1: Load the Audio Signals
# -----------------------------------------------------------------------------
# We load three audio signals:
#   - s1: the first source signal.
#   - s2: the second source signal.
#   - m : the observed mixture signal.
#
# We set sr=None so that the audio is loaded at its original sampling rate.
s1, sr1 = librosa.load("source1.wav", sr=None)
s2, sr2 = librosa.load("source2.wav", sr=None)
m, sr_m = librosa.load("mixture.wav", sr=None)

# -----------------------------------------------------------------------------
# Step 2: Check Sampling Rates
# -----------------------------------------------------------------------------
# For the linear combination to be valid, the source signals and the mixture must
# have the same sampling rate.
if sr1 != sr2 or sr1 != sr_m:
    raise ValueError("Sampling rates of the audio signals do not match!")

# -----------------------------------------------------------------------------
# Step 3: Prepare the Data Matrix
# -----------------------------------------------------------------------------
# Assume that the signals have the same number of samples (if not, they should be trimmed
# or padded appropriately). Let N be the number of samples.
N = len(m)

# Create matrix S with shape (N, 2) where:
#   - The first column is s1.
#   - The second column is s2.
#
# This formulation allows us to write the linear combination as:
#    m_hat = S w
# where w is a column vector containing the coefficients [a, b]^T.
S = np.column_stack((s1, s2))

# -----------------------------------------------------------------------------
# Step 4: Formulate the Least-Squares Problem
# -----------------------------------------------------------------------------
# Our goal is to find the coefficients a and b (contained in vector w) such that the
# reconstructed mixture:
#
#    m_hat = a * s1 + b * s2 = S w
#
# is as close as possible to the original mixture m in the least-squares sense.
#
# We define the cost function (or error function) as the squared error:
#
#    J(w) = || m - S w ||^2
#
# Expanding this squared norm:
#
#    J(w) = (m - S w)^T (m - S w)
#
# This expands to:
#
#    J(w) = m^T m - 2 w^T S^T m + w^T S^T S w
#
# The goal is to minimize J(w) with respect to w.

# -----------------------------------------------------------------------------
# Step 5: Derive the Normal Equation
# -----------------------------------------------------------------------------
# To minimize J(w), we take the derivative with respect to w and set it to zero.
#
# Compute the gradient:
#
#    ∇_w J(w) = -2 S^T m + 2 S^T S w
#
# Setting the gradient to zero:
#
#    -2 S^T m + 2 S^T S w = 0
#
# Divide both sides by 2:
#
#    S^T S w = S^T m
#
# This equation is known as the "normal equation."
# The solution is then given by:
#
#    w = (S^T S)^{-1} S^T m
#
# This solution gives the optimal coefficients a and b that minimize the cost function.

# -----------------------------------------------------------------------------
# Step 6: Solve the Least-Squares Problem
# -----------------------------------------------------------------------------
# We can solve for w using NumPy's least squares solver. This function computes the solution
# to the least-squares problem in a numerically stable manner.
w, residuals, rank, s = np.linalg.lstsq(S, m, rcond=None)
a, b = w  # Extract the coefficients

# Alternatively, if S^T S is invertible, one could use:
# w = np.linalg.inv(S.T @ S) @ (S.T @ m)
# but np.linalg.lstsq is more robust to numerical issues.

# -----------------------------------------------------------------------------
# Step 7: Reconstruct the Mixture and Evaluate
# -----------------------------------------------------------------------------
# Reconstruct the mixture using the optimal coefficients:
m_hat = a * s1 + b * s2

# Compute the mean squared error (MSE) between the original mixture and the reconstructed mixture.
mse = np.mean((m - m_hat) ** 2)

# -----------------------------------------------------------------------------
# Step 8: Print the Results
# -----------------------------------------------------------------------------
print("Optimal coefficients:")
print("  a =", a)
print("  b =", b)
print("Mean Squared Error of the reconstruction:", mse)
