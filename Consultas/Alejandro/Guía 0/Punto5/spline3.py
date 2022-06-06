# La función "spline3" permite obtener los coeficientes de los polinomios que conforman una spline cúbica. SON NO CANÓNICOS.
# Se emplean "n" puntos y esto da lugar a "n-1" polinomios de grado 3, donde cada uno posee 4 coeficientes. Entonces, en total hay 
# "4*(n-1)" coeficientes por calcular, entre los cuales hay "n-1 cúbicos" (almacenados en el vector "a"), "n-1 cuadráticos" (almacenados en el vector "b"), "n-1 lineales" (almacenados en el vector "c") y "n-1 constantes" (almacenados en el vector "d").
# El i-ésimo polinomio tiene la forma:
# f_i(x) = a[i] (x-x_i)^3 + b[i] (x-x_i)^2 + c[i] (x-x_i) + d[i]
# Requiere el vector "X" con abscisas y el vector "Y" con ordenadas.
# Además, se requiere la variable "tipoconds" que valdrá:
    # {=1} CURVATURAS FIJAS, es decir, condiciones aplicadas en la derivada segunda. Ellas estan almacenadas en el vector "cond"
    # (requerido por esta función) y son: "cond[0]=f''(X[0])" Y "cond[1]=f''(X[n-1])".
    # {=1} SPLINE NATURAL, que es un caso particular del anterior donde se verifica que "cond[0]=cond[1]=0".
    # {=2} BORDES FIJOS, es decir, condiciones aplicadas en la derivada primera. Ellas estan almacenadas en el vector "cond" y son:
    # "cond[0]=f'(X[0])" Y "cond[1]=f'(X[n-1])".
# La función devuelve la matriz "coeftes" de tamaño "(n-1)*4", que contiene a los vectores "a", "b", "c" y "d" en las columnas "1", "2", "3" y "4" respectivamente.

import numpy as np   # Importo paquete de álgebra lineal

def spline3(X,Y,tipoconds,cond):
    # Obtengo la cantidad de puntos "n" a interpolar.
    n = len(X)
    
    # Los "n" puntos dividen la abscisa en "n-1" intervalos, determinados por puntos contiguos "X[i]" y "X[i+1]", y cuya
    # longitud es "h[i] = X[i+1] - X[i]".
    # Defino vector "h" que contendrá la longitud de cada intervalo.
    h = []

    # Completo "h", resultará que tiene "n-1" componentes.
    for i in range(n-1):   # Va de "i = 0" a "i = n-2"
        h.append(X[i+1]-X[i])
    
    # Se buscan polinomios de la forma:
    # f_i(x) = a[i] (x-x_i)^3 + b[i] (x-x_i)^2 + c[i] (x-x_i) + d[i]

    # Se interpolará con un polinomio como el anterior, a cada uno de los "n-1" intervalos. Es decir, se tendrán "n-1" valores de
    # "a", "b", "c" y "d" (TENER ESTO EN CUENTA para las futuras definiciones de vectores "a", "b", "c" y "d").

    # Para que se reproduzca el conjunto experimental, se define:
    # d[i] = T[i]

    # Imponiendo continuidad en la función, se obtiene:
    # c[i] = ((T[i+1]-T[i])/h[i])) - (b[i]*h[i]) - (a[i]*(h[i]**2))

    # Imponiendo continuidad en la derivada segunda, se obtiene:
    # a[i] = ((1/3)*(b[i+1]-b[i])/h[i])

    # Se continúa según las condiciones de contorno a aplicar.
    if tipoconds == 1:
        # Se consideran condiciones de contorno aplicadas en la DERIVADA SEGUNDA, "cond[0]=f''(X[0])" Y "cond[1]=f''(X[n-1])".
        # Esto implica definir "b[0] = cond[0]" y "b[n-1] = cond[1]". VER que hice que hayan "n" valores de "b", y no "n-1" como dije
        # antes; es necesario agregar este valor n-ésimo adicional por el momento, pero MÁS ADELANTE LO BORRARÉ).

        # Para obtener un vector "b" que contemple las condiciones anteriores y además contenga los coeficientes "b" de los polinomios
        # interpolantes, se plantea un sistema de ecuaciones lineales "Ab=y" con "A" de tamaño "n*n" e "y" de tamaño "n*1".
        A = np.eye((n))
        y = np.zeros((n,1))
        y[0] = cond[0]     # Se definió así para obtener "1*b[0]=(cond[0])".
        y[n-1] = cond[1]   # Se definió así para obtener "1*b[n-1]=(cond[1])".
    
    elif tipoconds == 2:
        # Se consideran condiciones de contorno aplicadas en la DERIVADA PRIMERA, "cond[0]=f'(X[0])" Y "cond[1]=f'(X[n-1])".

        # Para obtener un vector "b" que contemple las condiciones anteriores y además contenga los coeficientes "b" de los polinomios
        # interpolantes, se plantea un sistema de ecuaciones lineales "Ab=y" con "A" de tamaño "n*n" e "y" de tamaño "n*1".
        A = np.zeros((n,n))
        A[0,0] = 2*h[0]
        A[0,1] = h[0]
        A[n-1,n-1] = 2*h[n-2]
        A[n-1,n-2] = h[n-2]
        y = np.zeros((n,1))
        y[0] = 3*(((Y[1]-Y[0])/h[0])-cond[0])    # Se definió así para obtener "1*b[0]=3*(((Y[1]-Y[0])/h[0])-cond[0])".
        y[n-1] = 3*(cond[1]-((Y[n-1]-Y[n-2])/h[n-2]))   # Se definió así para obtener "1*b[n-1]=3*(cond[1]-((Y[n-1]-Y[n-2])/h[n-2]))".
        
    # Debido a que "y[0]", "y[n-1]", "A[0,0]" y "A[n-1,n-1]" tienen su valor correcto, deberá recorrerse desde "i=1" hasta "i=n-2"
    # para asignar los valores correctos al resto de componentes de "A" e "y".
    for i in range(1,n-1):
        A[i,i-1]=h[i-1]
        A[i,i]=2*(h[i]+h[i-1])
        A[i,i+1]=h[i]
        y[i]=3*(((Y[i+1]-Y[i])/h[i])-((Y[i]-Y[i-1])/h[i-1]))
    
    # Obtengo "b".
    b = np.linalg.solve(A,y)
    
    # Encontrado "b", se pueden calcular por recursividad "a" y "c". Defino "a", "c" y "d".
    a = []
    c = []
    d = []

    # Completo "a", "c" y "d" empleando las expresiones presentadas más arriba. Estos vectores deben tener "n-1" componentes,
    # porque esa es la cantidad de intervalos y por ende la cantidad de polinomios interpolantes. "b" tiene "n" elementos hasta
    # el momento, por eso tenemos que plantear el FOR así (para que vaya desde "0" a "n-2").
    for i in range(len(b)-1):
        a.append(((1/3)*(b[i+1]-b[i])/h[i]))
        c.append((((Y[i+1]-Y[i])/h[i]) - b[i]*h[i] - a[i]*(h[i]**2)))
        d.append((Y[i]))
        
    # Ahora que ya completé los vectores "a", "c" y "d", convierto "b" en vector de "n-1" componentes.
    b = b[0:len(b)-1]
    
    # Defino la matriz "coeftes" que contiene a los vectores "a", "b", "c" y "d" en las columnas "1", "2", "3" y "4",
    # respectivamente.
    coeftes=np.zeros((n-1,4));

    # La completo recorriendo sus "n-1" filas.
    for i in range (n-1):
        coeftes[i,0]=a[i]
        coeftes[i,1]=b[i]
        coeftes[i,2]=c[i]
        coeftes[i,3]=d[i]
    
    return coeftes