import util
import numpy as np

data = util.load_dataset()
print(f"Total samples: {len(data)}")

depths = []
for i, sample in enumerate(data):
    try:
        _, _, Z = util.dataset_sample_to_images_and_depth(sample)
        depths.append(Z)
        if i < 10:
            print(f"Sample {i}: Z={Z:.3f}m")
    except Exception as e:
        print(f"Sample {i}: Error - {e}")

print(f"\nDepth distribution:")
print(f"  Min: {min(depths):.3f}m")
print(f"  Max: {max(depths):.3f}m")
print(f"  Unique values: {len(set(np.round(depths, 3)))}")