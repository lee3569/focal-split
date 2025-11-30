import pickle
import constants as const

def inspect_keys():
    with open(const.DATASET_PKL, "rb") as f:
        data = pickle.load(f)
    
    sample = data[0]    #call for first sample
    

    print(f"Item 0 (Type: {type(sample[0])}) Keys: {list(sample[0].keys())}")
    print(f"Item 1 (Type: {type(sample[1])}) Keys: {list(sample[1].keys())}")

if __name__ == "__main__":
    inspect_keys()