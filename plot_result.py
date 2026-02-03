import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 파일 읽기
filename = "problem1.dat"
data_blocks = []
current_block = {"n": None, "M": None, "R": None, "data": []}

try:
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split(',')))
        if len(parts) == 3: # 헤더: n, M, R
            if current_block["n"] is not None:
                data_blocks.append(current_block)
                current_block = {"n": None, "M": None, "R": None, "data": []}
            current_block["n"] = parts[0]
            current_block["M"] = parts[1]
            current_block["R"] = parts[2]
        elif len(parts) == 2: # 데이터: r, h
            current_block["data"].append(parts)
            
    if current_block["n"] is not None:
        data_blocks.append(current_block)

except FileNotFoundError:
    print("오류: problem1.dat 파일이 없습니다. 먼저 newtonian_solver.py를 실행하세요.")
    exit()

# 2. 그래프 설정
plt.figure(figsize=(10, 7))
K = 100.0 # 폴리트로픽 상수 (문제 조건)

# 3. 각 n에 대해 밀도 계산 및 플롯
for block in data_blocks:
    n = block['n']
    R_s = block['R'] # 물리적 반지름 (참고용)
    
    # 데이터 배열 변환
    data = np.array(block['data'])
    r_hat = data[:, 0] # 정규화된 반지름 (0 ~ 1)
    h = data[:, 1]     # 엔탈피
    
    # 밀도(rho) 계산: rho = (h / (K(n+1)))^n
    # h가 음수가 되면 0으로 처리 (표면 바깥)
    h_safe = np.maximum(h, 0)
    rho = (h_safe / (K * (n + 1))) ** n
    
    # 그래프 그리기
    # 문제 요구사항: x축은 r_hat, y축은 rho
    plt.plot(r_hat, rho, label=f"n = {n} (R={R_s:.2f})", linewidth=2)

# 4. 그래프 꾸미기
plt.xlabel(r"Normalized Radius $\hat{r}$ ($r/R$)", fontsize=14)
plt.ylabel(r"Density $\rho$", fontsize=14)
plt.title("Density Profile for Different Polytropic Indices", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1.0)

# 5. 파일 저장
plt.savefig("problem1_2_density_profile.png", dpi=300)
print("그래프가 'problem1_2_density_profile.png'로 저장되었습니다.")
plt.show()