# Paper Section 3.1, 4.1
FOCAL_LENGTH = 30e-3  # f: lens focal length (30mm from prototype)
APERTURE = 5e-3  # A: Gaussian aperture std (manual choice)

PIXEL_PITCH = 1.4e-6  # OV5647 sensor pixel pitch
SENSOR_WIDTH = 1920
SENSOR_HEIGHT = 1080

# Luo Paper Fig 1b: sensor distance
S1 = 35.0e-3  # s1: sensor distance for I1 (manual choice)
S2 = 35.4e-3  # s2: sensor distance for I2 (논문에서 0.4mm 차이)
S_CONSENSUS = (S1 + S2) / 2  # c: consensus sensor location (Eq. 6)