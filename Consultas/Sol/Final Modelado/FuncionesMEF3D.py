import numpy as np
#Para sacar los menores de la matriz y calcular los determinantes:
def Menores_Det(SeisV, i, j):
    Menor = np.delete(np.delete(SeisV,i,axis=0), j, axis=1)
    Det_Menor=np.abs(np.linalg.det(Menor))
    
    return Det_Menor

def Ensamblado_Matriz_Global(MN, MC, Me, glxn):

	Nn = MN.shape[0]  # Número de nodos
	Ne, Nnxe = MC.shape  # Ne: Número de elementos, # Número de nodos x elemento

	Matriz_Global = np.zeros([glxn * Nn, glxn * Nn])
	for element in range(Ne):
		for i in range(Nnxe):
			indices_i = np.linspace(i * glxn, (i + 1) * glxn - 1, Nnxe).astype(int)
			rangoni = np.linspace(MC[element, i] * glxn, (MC[element, i] + 1) * glxn - 1, Nnxe).astype(int)
			for j in range(Nnxe):
				indices_j = np.linspace(j * glxn, (j + 1) * glxn - 1, Nnxe).astype(int)
				rangonj = np.linspace(MC[element, j] * glxn, (MC[element, j] + 1) * glxn - 1, Nnxe).astype(int)
				Matriz_Global[np.ix_(rangoni, rangonj)] += Me[np.ix_(indices_i, indices_j)]

	return Matriz_Global

def solve(KGlobal, s,r, Us, Fr):
    N = KGlobal.shape[1]
    U = np.zeros(N)
    F = np.zeros(N)
    U[s] = Us
    F[r] = Fr
    Kred = KGlobal[np.ix_(r,r)]
    Kvin = KGlobal[np.ix_(r,s)]
    U[r] = np.linalg.solve(Kred, F[r]-Kvin.dot(U[s]))
    F[s] = KGlobal[s,:].dot(U)
    return U,F