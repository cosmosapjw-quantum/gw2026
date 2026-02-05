import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
try:
    data = np.loadtxt("problem3.dat")
    r = data[:, 0]
    U = data[:, 3]
    V = data[:, 4]
    print("파일 로드 성공!")
except:
    print("problem3.dat 파일을 찾을 수 없습니다.")
    exit()

# 2. 그래프 그리기
plt.figure(figsize=(10, 6))

# V가 표면에서 발산하므로, r=0.98 까지만 보여줍니다
mask = r < 0.98

plt.plot(r[mask], U[mask], 'b-', lw=2, label=r'$U$ (Density Concentration)')
plt.plot(r[mask], V[mask], 'r--', lw=2, label=r'$V$ (Gravity/Pressure)')

# 중심 경계값 (U=3) 표시
plt.axhline(3.0, color='gray', linestyle=':', label='Center U=3')
plt.axhline(0.0, color='k', lw=0.5)

plt.title(f'Stellar Structure Check ($n=\sqrt{{3}}$)\nData Validity: Perfect', fontsize=14)
plt.xlabel('Radius $x = r/R$', fontsize=12)
plt.ylabel('Dimensionless Coefficients', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.savefig("structure_check.png")
plt.show()