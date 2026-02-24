import pickle
import numpy as np

def debug_pkl_structure(file_path):
    print(f"\n{'='*50}")
    print(f"디버깅 시작: {file_path}")
    print(f"{'='*50}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"1. 최상위 데이터 타입: {type(data)}")
        
        # 리스트인 경우 첫 번째 요소 검사
        if isinstance(data, list):
            print(f"   리스트 길이: {len(data)}")
            sample = data[0]
            print(f"2. 첫 번째 요소(sample) 타입: {type(sample)}")
        else:
            sample = data
            
        # 딕셔너리인 경우 Key 확인
        if isinstance(sample, dict):
            print(f"3. 내부 Keys: {list(sample.keys())}")
            for key, val in sample.items():
                if isinstance(val, np.ndarray):
                    print(f"   - Key [{key}]: ndarray, Shape: {val.shape}, Dtype: {val.dtype}")
                else:
                    print(f"   - Key [{key}]: Type: {type(val)}, Value: {val}")
        
        # 넘파이 배열인 경우 (이미지가 그냥 리스트에 순서대로 담긴 경우)
        elif isinstance(sample, (list, np.ndarray)):
            print(f"3. 샘플 내부 요소 개수: {len(sample)}")
            for i, item in enumerate(sample):
                if isinstance(item, np.ndarray):
                    print(f"   - Index [{i}]: ndarray, Shape: {item.shape}, Dtype: {item.dtype}")
                else:
                    print(f"   - Index [{i}]: Type: {type(item)}, Value: {item}")
        
        # 실제 데이터 샘플 출력 (이미지 데이터인지 확인용)
        # 만약 Index 0번이 이미지라면 앞부분 5개 값 출력
        if isinstance(sample[0], np.ndarray):
             print(f"\n4. 데이터 샘플 (Index 0): \n{sample[0][:1, :5]}") # 첫 행의 앞 5개 값

    except Exception as e:
        print(f"에러 발생: {e}")

# 가장 의심되는 파일 하나만 먼저 확인
debug_pkl_structure("waterbottle.pkl")