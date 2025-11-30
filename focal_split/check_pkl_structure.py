import pickle
import numpy as np
import constants as const

def inspect_pickle():
    path = const.DATASET_PKL
    print(f"Loading: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    print(f"Data type: {type(data)}")
    print(f"Total samples: {len(data)}")
    
    sample = data[0]
    print(f"\n[Sample 0 Structure]")
    print(f"Type: {type(sample)}")
    
    if isinstance(sample, dict):
        print(f"Keys: {sample.keys()}")
        for k, v in sample.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                print(f" - Key '{k}': shape/len = {np.shape(v)}")
            else:
                print(f" - Key '{k}': value = {v}")
    elif isinstance(sample, (list, tuple)):
        for i, item in enumerate(sample):
            print(f" - Item {i}: Type={type(item)}, Shape={np.shape(item) if hasattr(item, 'shape') else 'scalar'}")

if __name__ == "__main__":
    inspect_pickle()