# Se ordenará "v" en forma ascendente, empleando método de BURBUJEO. Consiste en comparar cada par de elementos adyacentes 
# (v[j] con v[j+1]), comenzando desde el inicio del vector ("v[0]"). Si ambos no están ordenados (el segundo es menor que el 
# primero), se intercambian sus posiciones. En cada iteración, un elemento menos necesita ser evaluado (el último), ya que no 
# hay más elementos a su derecha que necesiten ser comparados, puesto que ya están ordenados.

import numpy as np   # Importo paquete de álgebra lineal

def ordburb(v):
    # Recorro "v" desde el inicio "i=0" hasta la penúltima componente (que es "i=len(v)-2" pero como python no toma límite
    # superior entonces debo colocar "i=len(v)-1").
    for i in range(0,len(v)-1):
        # Recorro "v" desde el inicio "j=0" hasta la penúltima componente "j=len(v)-2" menos "i", porque así voy evaluando un 
        # elemento menos por cada iteración, ya que el último fue ordenado en la iteración anterior.
        for j in range (0,len(v)-1-i):
            # Verifico si los elementos "v[j]" y "v[j+1]" están ordenados de manera ascendente, sino, intercambio.
            if v[j]>v[j+1]:
                # Guardo en "v[j]" en variable auxiliar "aux"
                aux=v[j]
                v[j]=v[j+1]
                v[j+1]=aux
    return v