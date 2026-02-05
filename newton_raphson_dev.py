import numpy as np
from scipy.optimize import fsolve
from scipy.differentiate import jacobian

# 생성형 AI 도입 전 작성한 부분 및 도입 후 수정내역 포함

def nrbase(f, x0, delta):
    dfdx = (f(x0+delta)-f(x0-delta))/delta/2
    x1 = x0 - f(x0)/dfdx
    return x1

def g(x):
    return x**2 - 1

x0 = -10
delta = 0.1
x1 = nrbase(g,x0,delta)

while abs(x1-x0) > 1e-5:
    x0 = x1
    x1 = nrbase(g,x0,delta)

def nrmethod(f,x0,delta,eps):
    dfdx = (f(x0+delta)-f(x0-delta))/delta/2
    x1 = x0 - f(x0)/dfdx
    while abs(x1-x0) > eps:
        x0 = x1
        dfdx = (f(x0+delta)-f(x0-delta))/delta/2
        x1 = x0 - f(x0)/dfdx
    return x1

def h(x):
    return 2*x**3 -5* x**2 + 3*x + 5

def nrmethod2D_basic(f,x0,eps):
    jac = jacobian(f,x0).df
    x1 = x0 - np.linalg.inv(jac)@f(x0)
    while True:
        x0 = x1
        jac = jacobian(f,x0).df
        x1 = x0 - np.linalg.inv(jac)@f(x0)
        if (abs(x1-x0) < eps).all():
            break
    return x1

def func(x):
    x1, x2 = x
    return [x1**2-x2,x1-x2**2]

# ODE 모듈과 연결하기 위해 병렬평가 및 broadcasting 가능한 버전으로 AI에게 업그레이드 요구한 버전

def get_numerical_jacobian(f, x0, eps=1e-8):
    x0 = np.asarray(x0, dtype=float)
    n_dim = len(x0)
    fx = f(x0)
    jac = np.zeros((n_dim, n_dim))
    
    for i in range(n_dim):
        x_perturbed = x0.copy()
        x_perturbed[i] += eps
        fx_perturbed = f(x_perturbed)
        jac[:, i] = (fx_perturbed - fx) / eps
    return jac

def nrmethod2D(f,x0,eps):
    x0 = np.asarray(x0, dtype=float)
    jac0 = get_numerical_jacobian(f, x0)
    fx0  = np.asarray(f(x0), dtype=float)
    x1   = x0 - np.linalg.solve(jac0, fx0)

    while True:
        x0 = x1
        jac1 = get_numerical_jacobian(f, x0)
        fx1  = np.asarray(f(x0), dtype=float)
        x1   = x0 - np.linalg.solve(jac1, fx1)
        if (np.abs(x1 - x0) < eps).all():
            break
    return x1

if __name__ == "__main__":

    realsol = - 1.0
    print("analytic solution과 비교한 1차원 버전 오차: ",x1-realsol)

    mysol = nrmethod(h,100.0,1e-5,1e-5)
    realsol = fsolve(h,100.0,xtol=1e-5)
    print("1차원 뉴턴-랩슨 방법 오차 : ",mysol-realsol)

    print("2차원 버전 뉴턴-랩슨 방법 오차 : ",fsolve(func,[0.3,1.2])-nrmethod2D_basic(func,[0.3,1.2],1e-4))