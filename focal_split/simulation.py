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


def simulation(target_size=512):
    
    pinhole_image = imaging.load_pinhole_image('focal_split/wood.png')
    
    
    h, w = pinhole_image.shape
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pinhole_image = cv2.resize(pinhole_image, (new_w, new_h))
        print(f"Resized: ({h}, {w}) -> ({new_h}, {new_w})")

    true_depth_z = 0.8

    psf1 = optics.create_psf(const.APERTURE, const.FOCAL_LENGTH, const.S1, true_depth_z, const.PIXEL_PITCH)
    psf2 = optics.create_psf(const.APERTURE, const.FOCAL_LENGTH, const.S2, true_depth_z, const.PIXEL_PITCH)

    print("Generating defocused images...")
    I1 = simul.generate_defocused_image(pinhole_image, psf1)
    I2 = simul.generate_defocused_image(pinhole_image, psf2)

    I1_aligned, I2_aligned = util.align_images(I1, I2)

    Is = oper.calculate_Is(I1_aligned, I2_aligned)
    laplacian_I = oper.calculate_laplacian(I1_aligned, I2_aligned)    

    a, b = optics.get_optical_constants_ab(const.APERTURE, const.FOCAL_LENGTH, const.S_CONSENSUS)

    depth_map = depth.calculate_depth_map(Is, laplacian_I, a, b)


    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(I1, cmap='gray')
    plt.title('I1 (defocused)')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(I2, cmap='gray')
    plt.title('I2 (defocused)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth (m)')
    plt.title(f'Depth Map (True: {true_depth_z}m)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulation(target_size=512)