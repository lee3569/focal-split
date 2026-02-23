import numpy as np
from scipy.optimize import minimize
import util
import imaging
import oper
import constants as const

def get_slope_error(params, data_samples):
    """
    [목표 함수]
    예측값들의 추세선 기울기를 구해서, 1.0과의 차이를 반환함.
    목표: 기울기(Slope)가 1.0이 되도록 A, B를 조절해라!
    """
    A_curr, B_curr = params
    
    # 0이나 음수가 들어오면 계산 터지니까 방지
    if A_curr <= 0 or B_curr <= 0: return 99999.0

    all_true = []
    all_pred = []
    
    for sample in data_samples:
        try:
            I1_rgb, I2_rgb, Ztrue = util.dataset_sample_to_images_and_depth(sample)
            
            # 전처리
            I1 = imaging.to_gray(I1_rgb)
            I2 = imaging.to_gray(I2_rgb)
            I1c, I2c = util.align_images(I1, I2)
            I1c = imaging.highpass_filter(I1c)
            I2c = imaging.highpass_filter(I2c)
            
            # 계산
            lap_I, It = oper.compute_laplacian_and_It(I1c, I2c)
            denom = A_curr * lap_I + B_curr * It
            depth_map = np.divide(lap_I, denom + 1e-10)
            
            # 중앙값
            h, w = depth_map.shape
            patch = depth_map[h//2-20:h//2+20, w//2-20:w//2+20]
            Z_pred = float(np.median(patch))
            
            # 유효한 범위만 수집
            if 0.0 < Z_pred < 5.0:
                all_true.append(Ztrue)
                all_pred.append(Z_pred)
        except:
            continue
            
    if len(all_true) < 10:
        return 99999.0 # 데이터 너무 적으면 패스

    # --- [핵심] 기울기(Slope) 계산 ---
    # np.polyfit(x, y, 1) -> [기울기, 절편]
    slope, intercept = np.polyfit(all_true, all_pred, 1)
    
    # 목표: 기울기는 1.0이어야 하고, 절편은 0.0이어야 함
    # Slope 에러에 가중치를 많이 줌 (x 100)
    loss = abs(slope - 1.0) * 100 + abs(intercept)
    
    return loss

def run_slope_optimization():
    print("Loading dataset...")
    data = util.load_dataset()
    data_subset = data[:50] # 50개 샘플 사용
    
    # 초기값 (현재 하드코딩된 값 근처에서 시작)
    initial_guess = [1.0, 0.8] 
    
    print(f"Optimizing for SLOPE = 1.0 (Initial: {initial_guess})...")
    
    result = minimize(
        get_slope_error, 
        initial_guess, 
        args=(data_subset,),
        method='Nelder-Mead',
        tol=1e-4
    )
    
    best_A, best_B = result.x
    print("\n" + "="*40)
    print("Slope Optimization Completed!")
    print("="*40)
    print(f"Optimal A = {best_A:.6f}")
    print(f"Optimal B = {best_B:.6f}")
    
    # 검증: 이 값으로 기울기가 진짜 1이 나오는지 확인
    final_loss = get_slope_error([best_A, best_B], data_subset)
    # loss = abs(slope - 1)*100 + abs(intercept) 였으므로 역산해보면 대략 알 수 있음
    print(f"Final Loss score: {final_loss:.4f}")
    print("-" * 40)
    
    # 자동 업데이트 여부
    print(f"[Reference] Constants.py: A={const.A_CALIB}, B={const.B_CALIB}")

    # 파일 업데이트 함수 (필요하면 주석 해제)
    # update_constants_file(best_A, best_B)

def update_constants_file(new_A, new_B):
    file_path = 'constants.py'
    with open(file_path, 'r') as f: lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines:
            if line.strip().startswith('A_CALIB'):
                f.write(f"A_CALIB = {new_A:.6f}\n")
            elif line.strip().startswith('B_CALIB'):
                f.write(f"B_CALIB = {new_B:.6f}\n")
            else:
                f.write(line)
    print("Updated constants.py with new values.")

if __name__ == "__main__":
    run_slope_optimization()