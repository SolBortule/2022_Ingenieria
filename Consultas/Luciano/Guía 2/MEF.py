import numpy as np

def solve1D(K, r, s, Us, Fr):
    """
    #
    Se deben ingresar:
    K = matriz global de constantes elásticas
    r = posicion de los desplazamientos desconocidos (fuerzas conocidas)
    s = posición de los desplazamientos conocidos 
    Us = valor de los desplazamientos conocidos 
    Fr = valor de las fuerzas conocidas
    
    mef.solve devuelve 2 vectores, uno con las fuerzas en cada nodo (F) y otro con los desplazamientos respectivamente (U)
    #
    """
    N=np.shape(K)[1]  # Nro. de nodos
    F = np.zeros([N,1])
    U = np.zeros([N,1])
    U[s] = Us
    F[r] = Fr
    Kred = K[np.ix_(r,r)]
    Kvin = K[np.ix_(r,s)]
    
    # Los 'r' son los desplazamientos incognitos, los 's' son los desplazamientos que tenemos como dato
    
    U[r] = np.linalg.solve(Kred, F[r]-Kvin.dot(U[s]))
    F[s] = K[s,:].dot(U)
    
    return F, U

def Kel_barra(MN, MC, Ee, Ae, e):
    """
    #
    Resuelve la matriz K_elemental de un elemento 'e'
    MN = Coordenadas de cada nodo
    MC = Matriz de conectividad de las barras
    Ee = Modulo de elasticidad de 'e'
    Ae = Sección de 'e'
    e = Nro. de elemento
    #
    """ 
    Le = np.sqrt((MN[MC[e,1],0]-MN[MC[e,0],0])**2+(MN[MC[e,1],1]-MN[MC[e,0],1])**2)
    phi = np.arctan2(MN[MC[e,1],1]-MN[MC[e,0],1],MN[MC[e,1],0]-MN[MC[e,0],0])
    ke = Ee*Ae/Le
    c = np.cos(phi)
    s = np.sin(phi)
    Ke = ke*np.array([[c**2,c*s,-c**2,-c*s],
                   [c*s,s**2,-c*s,-s**2],
                   [-c**2,-c*s,c**2,c*s],
                   [-c*s,-s**2,c*s,s**2]])
    Ke[np.abs(Ke/Ke.max()) < 1e-15] = 0

    return Ke
    
def Kglobal_barra(MN, MC, E, A, glxn):
    """
    #
    Resuelve la matriz global K
    MN = Coordenadas de cada nodo
    MC = Matriz de conectividad de las barras
    E = Vector Modulos de Elasticidad de cada elementos
    A = Vector Sección de cada elemento

    #
    """
    Ke = {}  # diccionario para acumular todas las K_elementales
    Nn = MN.shape[0]  # cantidad de nodos
    Ne = MC.shape[0]  # cantidad de elementos
    K = np.zeros([glxn*Nn,glxn*Nn])
    archivo = 'Matrices.txt'  # Creo archivo para guardar todas las matrices elementales
    with open(archivo,'w') as f:  # la f es como un alias apra el archivo que acabo de abrir
        f.write('Matrices Elementales\n ===============')

    for e in range(Ne):
        Ke[e] = Kel_barra(MN, MC, E[e], A[e], e)
        fe = np.abs(Ke[e].max())
        with open('Matrices.txt','a') as f:  # 'a' de agregar
            f.write(f'\nelemento {e}, fe ={fe:4e}\n')
            f.write(f'{Ke[e]/fe}\n')

        for i in range(glxn):
            rangoi = np.linspace(i*glxn,(i+1)*glxn-1,glxn).astype(int)
            rangoni = np.linspace(MC[e,i]*glxn,(MC[e, i]+1)*glxn-1,glxn).astype(int)
            for j in range(glxn):
                rangoj = np.linspace(j*glxn,(j+1)*glxn-1,glxn).astype(int)
                rangonj = np.linspace(MC[e,j]*glxn,(MC[e, j]+1)*glxn-1,glxn).astype(int)

                K[np.ix_(rangoni,rangonj)] += Ke[e][np.ix_(rangoi,rangoj)]
    
    fe = np.abs(K.max())
    with open('Matrices.txt','a') as f:  # 'a' de agregar
        f.write(f'\nMatriz Global, fe ={fe:4e}\n')
        f.write(f'{K/fe}\n')
            
    return K, Ke

def vector_complemento(s, MN, glxn):
    r = np.array([i for i in range(glxn*MN.shape[0]) if i not in s])
    return r