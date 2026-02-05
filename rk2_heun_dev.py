import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from newton_raphson_dev import nrmethod2D

# 생성형 AI 도입 전 작성한 부분 및 도입 후 수정내역 포함

def rkheun(f,y1,ini,fin,h):
    interval=np.linspace(ini,fin,int(1/h))
    for i in range(len(interval)):
        k1=f(interval[i],y1)
        k2=f(interval[i]+h,y1+h*k1)
        y1=y1+(k1+k2)*h/2
    return y1

def oderhs(x,y):
    return y

# np.empty(0)와 np.append를 넣어서 돌렸더니 너무 느려서 AI에게 물어보았고.
# 해당 코드는 O(n^2) 복잡도를 지니고 미리 array allocation을 하면 O(n)으로 줄어든다고 답함.

def rkheun2D(f,y0,ydot0,ini,fin,n):

    h = (fin-ini)/n
    t=np.linspace(ini,fin,n)

    y=np.empty(n)
    ydot=np.empty(n)
    
    y[0], ydot[0] = y0, ydot0
    for i in range(n-1):
        k1=f(t[i],y[i],ydot[i])
        k2=f(t[i+1],y[i]+h*ydot[i],ydot[i]+h*k1)
        y[i+1]=y[i] + h*ydot[i] + 0.5 * h**2 * k1
        ydot[i+1]=ydot[i] + 0.5*h*(k1+k2)

    return t, y, ydot

def yddot(t,y,ydot):
    return - y

def rkheunPoisson_basic(phiddot,hdot,phi_ini,h_ini,t_ini,t_fin,point,Rs):

    step = (t_fin-t_ini)/point
    t=np.linspace(t_ini,t_fin,point)

    phi=np.empty(point)
    phidot=np.empty(point)

    h=np.empty(point)
    
    phi[0], phidot[0] = phi_ini
    h[0] = h_ini

    for i in range(point-1):
        k1phi=phiddot(t[i],phi[i],phidot[i],h[i],Rs)
        k1h=hdot(t[i],phi[i],phidot[i],h[i],Rs)
        k2phi=phiddot(t[i+1],phi[i]+step*phidot[i],phidot[i]+step*k1phi,h[i]+step*k1h,Rs)
        k2h=hdot(t[i+1],phi[i]+step*phidot[i],phidot[i]+step*k1phi,h[i]+step*k1h,Rs)

        phi[i+1]=phi[i] + step*phidot[i] + 0.5 * step**2 * k1phi
        phidot[i+1]=phidot[i] + 0.5*step*(k1phi+k2phi)
        h[i+1]=h[i] + 0.5*step*(k1h+k2h)
       
    return t, phi, phidot, h

# 다음 버전은 broadcasting 및 뉴턴-랩슨 방법과 연동 가능하도록 AI에게 업그레이드 요구한 버전

def rkheunPoisson(phiddot,hdot,phi_ini,h_ini,t_ini,t_fin,point,Rs):

    step = (t_fin-t_ini)/point
    t=np.linspace(t_ini,t_fin,point+1)

    Rs = np.asarray(Rs, dtype=float)
    phi    = np.asarray(phi_ini[0], dtype=float)
    phidot = np.asarray(phi_ini[1], dtype=float)
    h      = np.asarray(h_ini, dtype=float)

    phi, phidot, h, Rs = np.broadcast_arrays(phi, phidot, h, Rs)

    for i in range(point-1):
        k1phi=phiddot(t[i],phi,phidot,h,Rs)
        k1h=hdot(t[i],phi,phidot,h,Rs)
        k2phi=phiddot(t[i+1],phi+step*phidot,phidot+step*k1phi,h+step*k1h,Rs)
        k2h=hdot(t[i+1],phi+step*phidot,phidot+step*k1phi,h+step*k1h,Rs)

        phi=phi + step*phidot + 0.5 * step**2 * k1phi
        phidot=phidot + 0.5*step*(k1phi+k2phi)
        h=h + 0.5*step*(k1h+k2h)

    return t, phi, phidot, h

K = 1.0*1e2
n = 0.8
rhoc = 1.28*1e-3

def hdot(rhat,phi,phidot,h,Rs):
    return - phidot

def phiddot(rhat,phi,phidot,h,Rs):
    return -(2./rhat)*phidot + 4.*np.pi* Rs**2 * (h/(K*(n+1)))**n

if __name__ == "__main__":
    sol = solve_ivp(oderhs,[0,1],[1],'RK45')
    print("기초적인 1차원 버전 오차", sol.y[0][-1] - rkheun(oderhs,1,0,1,1e-5))

    t, y, ydot = rkheun2D(yddot,1.,0.,0.,10.,1000)
    plt.plot(t,y)
    plt.savefig("rkheun2D_output.png")
    print("기초적인 2차 ODE (SHO) 그림 생성 완료")

    h_c = K*(1+n)*rhoc**(1.0/n)

    rhat_ini = 1e-5
    rhat_fin = 0.5
    point = 500

    # 실제로 ode를 풀기 위해서는 phi(0)와 phi'(0)가 필요함
    # 현재 버전에서는 게이지 자유도를 이용해 phi(0)를 임의로 0으로 잡음
    phi_ini_1 = [0.0,0.0]

    rhat, phi, phidot, h = rkheunPoisson_basic(phiddot,hdot,phi_ini_1,h_c,rhat_ini,rhat_fin,point,1.0)

    plt.plot(rhat,phi)
    plt.savefig("rkheunPoisson_basic_output.png")
    print("기초적인 포아송 방정식 버전 그림 생성 완료")