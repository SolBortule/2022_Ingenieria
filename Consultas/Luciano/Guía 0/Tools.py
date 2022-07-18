import numpy as np
import copy
import matplotlib.pyplot as plt


def thomas(A, B, C, D):
    # Algoritmo de Thomas

    n = len(D)

    C[0] = C[0]/B[0]
    D[0] = D[0]/B[0]

    A = [0] + A
    C = C + [0]

    for i in range(1, n):
        C[i] = C[i]/(B[i]-A[i]*C[i-1])

    for i in range(1, n):
        D[i] = (D[i]-A[i]*D[i-1])/(B[i]-A[i]*C[i-1])

    s = np.zeros(n)
    s[n-1] = D[n-1]
    for i in range(n-2, -1, -1):
        s[i] = D[i] - C[i]*s[i+1]

    return s


def spline3(S, z, r0, rn):
    # Cada variable debe estar en una columna de la matriz
    # z = 2, Spline con curvatura (S'')
    # z = 1, Spline con bordes sujetos (S')
    # z = 0, Spline natural
    # r0, rn: Derivadas al ppio. y al final

    x = copy.copy(S[:, 0])
    y = copy.copy(S[:, 1])
    n = len(x) - 1
    h = []
    d = []

    for i in range(n):
        h.append(x[i+1] - x[i])

    for i in range(n):
        d.append((y[i+1]-y[i])/h[i])

    if z == 0:
        z = 2
    # Spline z=0 o z=2 trabajan igual
    # Tengo s0 y sn que son r0 y rn, faltan los intermedios

    if z == 2:
        # Busco A, B, C
        # A = subdiagonal inferior
        # B = subdiagonal principal
        # C = subdiagonal superior
        A = []
        B = []
        C = []
        for i in range(n-2):
            A.append(h[i+1])
            C.append(h[i+1])

        for i in range(n-1):
            B.append(2*(h[i]+h[i+1]))

        D = []
        D.append(6*(d[1]-d[0])-h[0]*r0)
        for i in range(1, n-2):
            D.append(d[i+1]-d[i])
        D.append(6*(d[-1]-d[-2])-h[-1]*rn)

        s = thomas(A, B, C, D)
        s = np.insert(s, 0, r0)
        s = np.append(s, rn)

    elif z == 1:

        A = []
        B = []
        C = []

        for i in range(n-2):
            A.append(h[i+1])
            C.append(h[i+1])
        for i in range(n-1):
            B.append(2*(h[i]+h[i+1]))
        B[0] = 3/2*h[0] + 2*h[1]
        B[-1] = 2*h[-2] + 3/2*h[-1]

        D = []
        D.append(6*(d[1]-d[0])-3*(d[0]-r0))
        for i in range(1, n-2):
            D.append(6*(d[i+1]-d[i]))
        D.append(6*(d[-1]-d[-2])-3*(rn-d[-1]))

        s = thomas(A, B, C, D)
        s0 = 3/h[0]*(d[0]-r0)-1/2*s[0]
        sn = 3/h[-1]*(rn-d[-1])-1/2*s[-1]
        s = np.insert(s, 0, s0)
        s = np.append(s, sn)

    delta = []
    gamma = []
    beta = []
    alfa = []

    for i in range(n):
        delta.append(y[i])
        gamma.append(d[i]-((h[i]*(2*s[i]+s[i+1]))/6))
        beta.append(s[i]/2)
        alfa.append((s[i+1]-s[i])/(6*h[i]))
    coef = [alfa, beta, gamma, delta]
    return coef


def grspline3(S, coef, p, d=0):
    # p = presición para graficar
    # d = 0 gráfica f(x)
    # d = 1 gráfica f'(x)
    x = S[:, 0]
    n = np.size(coef, 1)  # cantidad de parabolas
    f = np.zeros([np.size(coef, 1), p])
    xa = []
    for i in range(n):  # habrá n-1 parabolas
        X = np.linspace(x[i], x[i+1], p)  # cantidad de puntos que forman cada parabola
        for j in range(p):
            f[i, j] = coef[0, i]*(X[j]-x[i])**3+coef[1, i]*(X[j]-x[i])**2+coef[2, i]*(X[j]-x[i])+coef[3, i]
        plt.plot(X, f[i, :])
        xa.append(X)
    if d == 1:
        fd = np.zeros([np.size(coef, 1), p])
        for i in range(n):
            X = np.linspace(x[i], x[i+1], p)
            for j in range(p):
                fd[i, j] = 3*coef[0, i]*(X[j]-x[i])**2+2*coef[1, i]*(X[j]-x[i])+coef[2, i]
            plt.plot(X, fd[i, :])

    if d == 1:
        return xa, fd, plt.show()
    elif d == 0:
        return xa, f, plt.show()