# Focal Split — Depth from Defocus

A Python implementation of the **Focal Split** depth-from-defocus (DfD) algorithm, based on the differential defocus framework. Given two images captured at slightly different focus settings, this pipeline estimates per-pixel depth from the differential blur between them.

---

## Background

Focal Split is grounded in the **defocus brightness constancy constraint** (Alexander, 2019), which relates small changes in camera focus parameters to spatial image derivatives under the Gaussian PSF assumption.

Given two images `I1`, `I2` captured at focus settings `s1`, `s2`, the blur kernel is modeled as:

$$
\sigma(Z) = A\left(\frac{1}{Z} - \rho\right)s + A
$$

Differentiating the image formation model `I(x; s) = k(x; s) * P(x)` with respect to `s` and applying the key Gaussian PSF property (`k_s = -(c²σA / s³) ∇²k`) yields a closed-form depth estimate:

$$
Z = \frac{a}{b + \tilde{I}_s / \nabla^2 \tilde{I}}
$$

where `a`, `b` are calibrated constants, `Ĩ_s` is the differential defocus term, and `∇²Ĩ` is the Laplacian of the reference image.

---

## Pipeline

```
I1, I2 (grayscale)
    │
    ▼
Geometric Alignment (SIFT + Homography)
    │
    ▼
Crop valid region (remove black borders post-warp)
    │
    ▼
Aberration Correction  →  removeLowFreqInfo (box filter, kernel=21)
    │
    ▼
Noise Attenuation  →  GaussianBlur (I1: 5×5, I2: 11×11)
    │
    ▼
Reference image:   I_avg = (I1 + I2) / 2
Laplacian term:    ∇²I  = I_avg − GaussianBlur(I_avg, 11×11)
Differential term: I_s  = (I1 − I2) / 2
    │
    ▼
Depth:  Z = V / W
        V = ∇²I
        W = A·∇²I + B·I_s       [A=1.1, B=0.3]
    │
    ▼
Depth map + Heatmap visualization
```

**Preprocessing order matters.** Aberration correction must precede noise attenuation, and geometric alignment must be applied to the full-resolution image before cropping.

---

## Key Implementation Notes

### SIFT-based Geometric Alignment
The two focal images have slight geometric misalignment due to the focus shift. Standard alignment is insufficient — SIFT feature detection with homography estimation is used to warp `I1` onto `I2`'s coordinate frame before any processing.

```python
I1_aligned, I2_aligned, H = util.align_images(I1_gray, I2_gray)
# Crop after warping to remove black border artifacts
I1_crop = I1_aligned[crop_region]
I2_crop = I2_aligned[crop_region]
```

This reframes what was originally an optical misalignment problem as a software-domain image registration problem.

### Aberration Correction
Low-frequency illumination bias is removed using a high-pass filter (image minus box-filtered version):

```python
I1_proc = removeLowFreqInfo(I1_crop, kernel_size=21)
I2_proc = removeLowFreqInfo(I2_crop, kernel_size=21)
```

### Noise Attenuation
Different kernel sizes are used per image to balance detail preservation vs. noise suppression:

```python
I1_proc = cv2.GaussianBlur(I1_proc, (5, 5), 0)   # preserves more detail
I2_proc = cv2.GaussianBlur(I2_proc, (11, 11), 0)  # stronger smoothing
```

### Depth Estimation

```python
I_avg      = (I1_proc + I2_proc) / 2
Laplacian  = I_avg - cv2.GaussianBlur(I_avg, (11, 11), 0)
I_s        = (I1_proc - I2_proc) / 2

V = Laplacian
W = 1.1 * Laplacian + 0.3 * I_s

Z = V / W
```

Parameters `A=1.1`, `B=0.3` were calibrated using a ground-truth ruler dataset heatmap.

---

## Results

Depth maps and heatmaps were validated against ground-truth ruler measurements. Predicted depth values increase monotonically with true depth across the tested range, confirming the algorithm's trend correctness. Absolute depth scale requires per-setup calibration of `A` and `B`.

> **Note:** This implementation is pixel-based, which preserves edge detail. This differs from aggregation-based approaches that group pixels before depth estimation, producing smoother, more planar outputs.

---

## Limitations

- Depth accuracy degrades away from the focal plane as defocus blur attenuates the image signal
- Relies on the Gaussian PSF assumption — real lenses deviate from this
- Per-pixel estimation is sensitive to noise; spatial aggregation (e.g., least-squares over local windows) improves stability
- Calibrated constants `A`, `B` are dataset-specific

---

## Dependencies

```
opencv-python
numpy
matplotlib
```

---

## References

- E. Alexander, "A theory of depth from differential defocus," Ph.D. dissertation, Harvard University, 2019.
- N. Persch et al., "Physically inspired depth-from-defocus," *Image and Vision Computing*, vol. 57, pp. 114–129, 2017.

---

## Related

This implementation is part of a broader study of depth-from-defocus methods in Guo Lab (Purdue University), covering Focal Track → Focal Split → Coupled Optical Differentiation (CoD, IJCV 2025).
