import os

# Paper Section 3.1, 4.1
FOCAL_LENGTH = 30e-3   # f: lens focal length (30mm from prototype)
APERTURE     = 8e-3    # A: Gaussian aperture std (manual choice)

PIXEL_PITCH  = 1.4e-6  # OV5647 sensor pixel pitch
SENSOR_WIDTH = 1920
SENSOR_HEIGHT = 1080

# Luo Paper Fig. 1(b): sensor distance
S1 = 33.0e-3     # s1: sensor distance for I1
S2 = 36.0e-3     # s2: sensor distance for I2 (0.4mm difference)
S_CONSENSUS = (S1 + S2) / 2.0   # c: consensus sensor location (Eq. 6)

# Calibrated depth coefficients (fitted to Eq. 11)
# Z(x) = a / ( b + I_s(x;s) / ∇²I(x;s) )
A_CALIB = 1.080831
B_CALIB = 0.815744

# Dataset path (Luo untethered snapshot dataset)
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PKL = os.path.join(ROOT_DIR, "saved_list_20250321_36d5_far.pkl")
