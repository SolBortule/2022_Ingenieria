import numpy as np
#Para sacar los menores de la matriz y calcular los determinantes:
#def Menores_Det(SeisV, i, j):
    #Menor = np.delete(np.delete(SeisV,i,axis=0), j, axis=1)
    #Det_Menor=np.linalg.det(Menor)
    #return Det_Menor
 #esta funcion tiene algo mal. Me daban mal algunos coeficientes. Lo calculo  a mano en la funcion k_elemental_3D

def Ensamblado_Matriz_Global(MN, MC, Me, glxn):

	Nn = MN.shape[0]  
	Ne, Nnxe = MC.shape 
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

def k_elemental_3D(MN, MC, nu, E, elemento):
    
    Fila_MC = MC[elemento, :].astype(int)
    x1, y1, z1= MN[Fila_MC[0],0], MN[Fila_MC[0],1], MN[Fila_MC[0],2]
    x2, y2, z2= MN[Fila_MC[1],0], MN[Fila_MC[1],1], MN[Fila_MC[1],2]
    x3, y3, z3= MN[Fila_MC[2],0], MN[Fila_MC[2],1], MN[Fila_MC[2],2]
    x4, y4, z4= MN[Fila_MC[3],0], MN[Fila_MC[3],1], MN[Fila_MC[3],2]
    
    SeisV = np.array([[1, x1, y1, z1],
                      [1, x2, y2, z2],
                      [1, x3, y3, z3],
                      [1, x4, y4, z4]])
    
    V=np.abs(np.linalg.det(SeisV)/6)
    # print('V: ', V*6)
    D=np.array([[1-nu,nu,nu,0,0,0],
       [nu,1-nu,nu,0,0,0],
       [nu,nu,1-nu,0,0,0],
       [0,0,0,(1-2*nu)/2,0,0],
       [0,0,0,0,(1-2*nu)/2,0],
       [0,0,0,0,0,(1-2*nu)/2]])*(E/((1+nu)*(1-2*nu)))
    
    #Coeficientes=np.zeros((4,4))
    #for i in range(4):
    #   for j in range(4):
     #       Coeficientes[i, j]=Menores_Det(SeisV, i, j)

    alpha1 = np.linalg.det(np.delete(np.delete(SeisV, 0, axis=0), 0, axis=1)) 
    #Con np.delete "elimino" temporalmente una fila  0 (Axis=0) y luego las columnas 0 1 2 y 3 (axis=1)
    beta1 = -np.linalg.det(np.delete(np.delete(SeisV, 0, axis=0), 1, axis=1))
    gamma1 = np.linalg.det(np.delete(np.delete(SeisV, 0, axis=0), 2, axis=1))
    delta1 = -np.linalg.det(np.delete(np.delete(SeisV, 0, axis=0), 3, axis=1))

    alpha2 = -np.linalg.det(np.delete(np.delete(SeisV, 1, axis=0), 0, axis=1))
    beta2 = np.linalg.det(np.delete(np.delete(SeisV, 1, axis=0), 1, axis=1))
    gamma2 = -np.linalg.det(np.delete(np.delete(SeisV, 1, axis=0), 2, axis=1))
    delta2 = np.linalg.det(np.delete(np.delete(SeisV, 1, axis=0), 3, axis=1))

    alpha3 = np.linalg.det(np.delete(np.delete(SeisV, 2, axis=0), 0, axis=1))
    beta3 = -np.linalg.det(np.delete(np.delete(SeisV, 2, axis=0), 1, axis=1))
    gamma3 = np.linalg.det(np.delete(np.delete(SeisV, 2, axis=0), 2, axis=1))
    delta3 = -np.linalg.det(np.delete(np.delete(SeisV, 2, axis=0), 3, axis=1))

    alpha4 = -np.linalg.det(np.delete(np.delete(SeisV, 3, axis=0), 0, axis=1))
    beta4 = np.linalg.det(np.delete(np.delete(SeisV, 3, axis=0), 1, axis=1))
    gamma4 = -np.linalg.det(np.delete(np.delete(SeisV, 3, axis=0), 2, axis=1))
    delta4 = np.linalg.det(np.delete(np.delete(SeisV, 3, axis=0), 3, axis=1))

    B1 = np.array([[beta1, 0, 0],
                 [0, gamma1, 0],
                 [0, 0, delta1],
                 [gamma1, beta1, 0],
                 [0, delta1, gamma1],
                 [delta1, 0, beta1]])
   
    
    B2 = np.array([[beta2, 0, 0],
                 [0, gamma2, 0],
                 [0, 0, delta2],
                 [gamma2, beta2, 0],
                 [0, delta2, gamma2],
                 [delta2, 0, beta2]])
   
    
    B3 = np.array([[beta3, 0, 0],
                 [0, gamma3, 0],
                 [0, 0, delta3],
                 [gamma3, beta3, 0],
                 [0, delta3, gamma3],
                 [delta3, 0, beta3]])
   
    B4 = np.array([[beta4, 0, 0],
                 [0, gamma4, 0],
                 [0, 0, delta4],
                 [gamma4, beta4, 0],
                 [0, delta4, gamma4],
                 [delta4, 0, beta4]])
   
    
    B = np.hstack([B1,B2,B3,B4])/(6*V)
    Kel = np.transpose(B).dot(D.dot(B))*V

    return Kel, D, B