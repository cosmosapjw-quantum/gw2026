import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import equations
import numerical

equations.K = 3.00
equations.n = np.sqrt(3)
numerical.K = 3.00
numerical.n = np.sqrt(3)

class Problem2StrictSolver(numerical.Numerical):
    def __init__(self):
        super().__init__()
        self.rhoc = 1.28e-3 #주어진 rhoc을 이용함
        
        # grid 간격 delta r = 1.0 x 10^-3 -> N = 1001
        self.N_grid = 1001 
        self.dr = 1.0e-3
        
        self.Rs = None
        self.Ms = None
        self.bg_data = None # [r, h, dh]

    def build_background(self):
        # 1. Shooting으로 Rs, Ms 결정 numerical.py 함수 사용
        Rs, Ms, res, ok = self.shooting(self.rhoc, 5.0, 1.0, point_main=self.N_grid, n_retry=2)
            
        self.Rs = Rs
        self.Ms = Ms
        print(f"   -> M={Ms:.5f}, R={Rs:.5f}") #Ms 와 Rs 값이 얼마인지 궁금해서 어떻게 표기하면 될지 AI을 사용했습니다.

        # 2.rkheunNewton_end 이용
        N_half = (self.N_grid - 1) // 2 # 500개 500개 나눠서 shooting
        
        # Inner: 0 -> 0.5
        # 중앙 초기값 설정하기
        eps = 1e-6
        a = (4*np.pi/3) * Rs**2 * self.rhoc
        phi = 0.5 * a * eps**2
        phidot = a * eps
        h = self.central_entalphy(self.rhoc) - phi
        
        r_in = np.linspace(0, 0.5, N_half + 1)
        h_list_in = [h]
        dh_list_in = [-phidot]
        
        curr_phi, curr_phidot, curr_h = phi, phidot, h
        for i in range(N_half):
            curr_phi, curr_phidot, curr_h = self.rkheunNewton_end(
                curr_phi, curr_phidot, curr_h, 
                r_in[i], r_in[i+1], point=1, Rs=Rs
            )
            h_list_in.append(curr_h)
            dh_list_in.append(-curr_phidot) # dh = -phidot

        # Outer: 1.0 -> 0.5
        # 초기값 설정
        phi = -Ms / Rs
        phidot = Ms / Rs
        h = 0.0
        
        r_out = np.linspace(1.0, 0.5, N_half + 1)
        h_list_out = [h]
        dh_list_out = [-phidot]
        
        curr_phi, curr_phidot, curr_h = phi, phidot, h
        for i in range(N_half):
            curr_phi, curr_phidot, curr_h = self.rkheunNewton_end(
                curr_phi, curr_phidot, curr_h, 
                r_out[i], r_out[i+1], point=1, Rs=Rs
            )
            h_list_out.append(curr_h)
            dh_list_out.append(-curr_phidot)
         # ---------------------------------------------------------
         # 이 부분은 계산이 느려지거나 제대로 계산이 되지 않아 AI을 이용해서 속도 최적화 코드를 만들었습니다
         # ---------------------------------------------------------   
        # 병합 (0 -> 1 정렬)
        self.bg_r = np.concatenate([r_in, r_out[:-1][::-1]])
        self.bg_h = np.concatenate([h_list_in, h_list_out[:-1][::-1]])
        self.bg_dh = np.concatenate([dh_list_in, dh_list_out[:-1][::-1]])

        # 보간용 데이터 저장 (속도 최적화)
        self.bg_data = np.column_stack((self.bg_h, self.bg_dh))

    def get_bg_idx(self, idx):
        return self.bg_data[idx]

    def oscillation_derivs(self, r, y, w2, h0, dh0):
        # 진동 미분방정식
        xi, dh = y
        
        # r=0이나 h=0으로 가지 않기 위해 0보다 조금 큰값을 대입함
        r_safe = max(r, 1e-6)
        h0_safe = max(h0, 1e-10)
        n = numerical.n
        
        dxi = -(3/r_safe)*xi - (n/(r_safe*h0_safe))*dh - (n/h0_safe)*dh0*xi
        ddh = (self.Rs**2) * w2 * r_safe * xi
        
        return np.array([dxi, ddh])

    def integrate_osc(self, w2, return_data=False):
        n_val = numerical.n
        N_half = (self.N_grid - 1) // 2 
        
    
        h_c = self.bg_data[0, 0]
        y_in = np.array([1.0, -3.0 * h_c / n_val])
        
        path_in = [y_in]
        curr = y_in
        
        for i in range(N_half):
            r_curr = self.bg_r[i]
            r_next = self.bg_r[i+1]
            h0, dh0 = self.get_bg_idx(i) 
            
            # RK2 Step
            dt = self.dr
            
            k1 = self.oscillation_derivs(r_curr, curr, w2, h0, dh0)
            
            h1, dh1 = self.get_bg_idx(i+1)
            y_pred = curr + dt * k1
            k2 = self.oscillation_derivs(r_next, y_pred, w2, h1, dh1)
            
            curr = curr + 0.5 * dt * (k1 + k2)
            path_in.append(curr)

         # ---------------------------------------------------------
         # 계산의 오류가 계속 발생해서 AI의 도움을 받아r=1.0 -> r=0.999 (Index 999) 구간은 테일러 급수(Analytic Step)로 처리했다.
         # ---------------------------------------------------------    
       
        idx_surf = self.N_grid - 1
        h_s, dh_s = self.get_bg_idx(idx_surf)
        d2h_s = -2 * dh_s
        
        xi_surf = 1.0
        dh_surf = -dh_s * xi_surf
        y_surf = np.array([xi_surf, dh_surf])
        
        # Taylor Step (r=1.0 -> r=0.999) 이 부분을 AI가 도와줌
        bracket = 3*dh_s + n_val*(self.Rs**2)*w2 + n_val*d2h_s
        dxi_ds = -(1/(n_val+1)) * (1/dh_s) * bracket
        ddh_ds = (self.Rs**2) * w2
        
        xi_999 = xi_surf - self.dr * dxi_ds
        dh_999 = dh_surf - self.dr * ddh_ds
        curr = np.array([xi_999, dh_999])
        
        path_out = [y_surf, curr]
        
        for i in range(N_half - 1):
            idx_curr = (self.N_grid - 2) - i
            idx_next = idx_curr - 1
            
            r_curr = self.bg_r[idx_curr]
            r_next = self.bg_r[idx_next]
            
            h0, dh0 = self.get_bg_idx(idx_curr)
            h1, dh1 = self.get_bg_idx(idx_next)
            
            dt = -self.dr # 역방향 적분
            
            k1 = self.oscillation_derivs(r_curr, curr, w2, h0, dh0)
            y_pred = curr + dt * k1
            k2 = self.oscillation_derivs(r_next, y_pred, w2, h1, dh1)
            
            curr = curr + 0.5 * dt * (k1 + k2)
            path_out.append(curr)
            
        # --- 3. Matching ---
        match_in = path_in[-1]   
        match_out = path_out[-1] 
        
        det = match_in[0]*match_out[1] - match_in[1]*match_out[0]
        
        if return_data:
            scale = match_in[0] / match_out[0]
            arr_in = np.array(path_in)
            arr_out = np.array(path_out)[::-1] # 뒤집어서 0.5 -> 1.0 순서로
            
            # Scale outer solution
            arr_out[:, 0] *= scale
            arr_out[:, 1] *= scale
            
            # Combine (r=0.5 중복 제거)
            full_xi = np.concatenate([arr_in[:, 0], arr_out[1:, 0]])
            full_dh = np.concatenate([arr_in[:, 1], arr_out[1:, 1]])
            
            return self.bg_r, full_xi, full_dh
            
        return det

    def find_modes(self):
        modes = []
        w_grid = np.concatenate([
            np.linspace(0.001, 0.2, 100),
            np.linspace(0.2, 5.0, 100)
        ])
        
        dets = [self.integrate_osc(w) for w in w_grid]
        
        for i in range(len(dets)-1):
            if dets[i] * dets[i+1] < 0:
                    root = brentq(self.integrate_osc, w_grid[i], w_grid[i+1])
                    modes.append(root)
                    if len(modes) >= 3: break
        return modes

if __name__ == "__main__":
    solver = Problem2StrictSolver()
    solver.build_background()
    modes = solver.find_modes()
    
        # ---------------------------------------------------------
         # 그래프 형식은 AI에 검색함
         # ---------------------------------------------------------    
    with open("problem2.dat", "w") as f:
        plt.figure(figsize=(10, 6))
        colors = ['b', 'r', 'g']
        
        for i, w2 in enumerate(modes):
            r, xi, dh = solver.integrate_osc(w2, return_data=True)
            omega = np.sqrt(w2)
            print("omega=", omega)
            
            f.write(f"{solver.Ms:.6e} {solver.Rs:.6e} {omega:.6e}\n")
            for rv, xv, dhv in zip(r, xi, dh):
                f.write(f"{rv:.6e} {xv:.6e} {dhv:.6e}\n")
                
            plt.plot(r, xi, label=f'Mode {i}', color=colors[i], lw=2)
            
    plt.title(f'Radial Oscillation Modes ($K=3.00, n=\sqrt{{3}}$)\nStrict Grid $\Delta r=10^{{-3}}$')
    plt.xlabel('$\hat{r}$')
    plt.ylabel(r'$\xi$')
    plt.axhline(0, c='k', lw=0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("problem2_FIG.png")
    plt.show()