import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import constants as const
from . import imaging
from . import simul
from . import util
from . import oper
from . import depth
from . import optics

def simulation():
    pinhole_image = imaging.load_pinhole_image('path/to/your/clear_image.png')

    depth_z = 0.8

    psf1 = optics.create_psf(const.APERTURE, const.FOCAL_LENGTH, const.S1, true_depth_z, const.PIXEL_PITCH)
    psf2 = optics.create_psf(const.APERTURE, const.FOCAL_LENGTH, const.S2, true_depth_z, const.PIXEL_PITCH)

    I1 = simul.generate_defocused_image(pinhole_image, psf1)
    I2 = simul.generate_defocused_image(pinhole_image, psf2)

    I1_aligned, I2_aligned = util.align_images(I1, I2)

    Is = oper.calculate_Is(I1_aligned, I2_aligned)
    laplacian_I = oper.calculate_laplacian(I1_aligned, I2_aligned)    

    a, b = optics.get_optical_constants_ab(const.APERTURE, const.FOCAL_LENGTH, const.S_CONSENSUS)

    depth_map = depth.calculate_depth_map(Is, laplacian_I, a, b)

    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar()
    plt.title(f"Estimated Depth Map (True Depth: {true_depth_z}m)")
    plt.show()

if __name__ == '__main__':
    run_simulation()