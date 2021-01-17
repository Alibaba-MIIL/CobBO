import numpy as np
import math

def rosenbrock(x):
    a = x[0:x.size - 1] ** 2 - x[1:x.size]
    b = x - 1
    return -(100 * np.inner(a, a) + np.inner(b, b))


def schwefel(x):
    fit = 0
    for i in range(x.size):
        fit += (x[0:i+1].sum()) ** 2
    return -fit


def ackley(x):
    return -(-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
        np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)


alpha = [1.00, 1.20, 3.00, 3.20]
A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
              [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
              [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
              [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091, 381]])
def hartmann(x):
    """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum = internal_sum + A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum = external_sum + alpha[i] * np.exp(-internal_sum)

    return -external_sum


def Levy(x):
    # ref: http://www.sfu.ca/~ssurjano/levy.html
    x = np.array(x)
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]

    s0 = (math.sin(math.pi * w[0])) ** 2

    s1 = 0.0
    d = x.size
    for i in range(d - 1):
        s1 += (((w[i] - 1.0) ** 2) * (1.0 + 10.0 * (math.sin(math.pi * w[i] + 1.0)) ** 2))

    s2 = ((w[d - 1] - 1.0) ** 2) * (1.0 + (math.sin(2.0 * math.pi * w[d - 1])) ** 2)

    return -(s0 + s1 + s2)


def Rastrigin(x):
    #ref: http://www.sfu.ca/~ssurjano/rastr.html
    x = np.array(x)
    sum = 10.0 * x.size
    for xi in x:
        sum += ( xi**2 - 10.0 * math.cos(2.0*math.pi * xi)  )
    return -sum


def Hartmann6D(x):
    # ref https://www.sfu.ca/~ssurjano/hart6.html
    alpha = [1.0, 1.2, 3.0, 3.2];
    A = [[10, 3, 17, 3.5, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]]

    # 10^(-4) *
    P = [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]]

    outer = 0.0
    for ii in range(4):
        inner = 0.0
        for jj in range(6):
            xj = x[jj]
            Aij = A[ii][jj]
            Pij = P[ii][jj] * 1.0e-4
            inner = inner + Aij * (xj - Pij) ** 2;
        new = alpha[ii] * np.exp(-inner);
        outer = outer + new;

    y = -(2.58 + outer) / 1.94;
    return -y
