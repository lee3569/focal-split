"""
Constants for Focal Split depth estimation
All working settings
"""

# Calibrated parameters
A_CALIB = 1.1   # Works well!
B_CALIB = 0.3   # Works well!

# Depth visualization range
DEPTH_MIN = 0.4  # 0.4~1.0m
DEPTH_MAX = 1.0

# Confidence threshold (normalized)
CONF_THRESHOLD = 0.15  # Normalized confidence > 0.15

# Physical parameters (from paper)
FOCAL_LENGTH = 0.030  # 30mm
APERTURE = 0.008      # 8mm
S1 = 0.033            # Sensor 1 position (33mm)
S2 = 0.036            # Sensor 2 position (36mm)
DELTA_S = S2 - S1     # 0.003m = 3mm

# Processing parameters
WINDOW_SIZE = 21      # Spatial aggregation window
HIGHPASS_SIZE = 21    # High-pass filter size
SMOOTH_SIZE = 11      # Gaussian blur for denoising
LAP_SIZE = 11         # Laplacian blur size

# Crop settings
CROP_DEFAULT = 50     # Border crop size