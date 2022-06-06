# La función evalúa el desarrollo de Taylor de "e^x" en el valor de entrada "x". Se obtiene un resultado con al menos "cs" cifras significativas correctas.

import numpy as np   # Importo paquete de álgebra lineal
import math   # Importo paquete de funciones matemáticas

def expy(x,cs):
    # Calculo el valor de referencia "teo".
    teo=math.exp(x)
    # Defino "vf = 0" para ir incrementando su valor a medida que agrego términos.
    vf = 0
    # Defino "i = 0" como valor de exponente para la potencia del primer término.
    i = 0
    # El valor final "vf" deberá tener al menos "cs" cifras significativas correctas con respecto a "teo", es decir:
    # teo-vf > 10^(-cs+1).
    # Por ejemplo, si "teo = 1,2345" y "cs = 3", debería obtener "vf = 1,233X", implica que debe cumplirse 1,2345-1,233x < 0,01,
    # porque si fuera >= no se estaría cumpliendo la cifra "cs" (daría 1,2245 si fuera =).
    while (teo-vf) >= 10**(-cs+1):
        # Debo sumar otro término. Actualizo "vf" usando "vf += algo" que es equivalente a poner "vf = vf + algo"
        vf += (x**(i))/math.factorial(i)
        # Incremento en 1 el valor del exponente, por si debo agregar otro término a la sumatoria.
        i += 1
    return vf