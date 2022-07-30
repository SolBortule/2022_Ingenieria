import numpy as np
import gmsh
import FuncionesMEF3D as M3D

gmsh.initialize()

gmsh.model.add('Medio Arco Recurvo')

lc = 10
E = 32e6  # Pa???????
nu = 0.3
GLXN = 3

# Defino los puntos iniciales del modelo
Posicion_Traccionado_Y_superior = 80
Posicion_Traccionado_Y_inferior = 78
p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(16, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(14, 10, 0, lc)
p4 = gmsh.model.occ.addPoint(0, Posicion_Traccionado_Y_inferior, 0, lc)
p5 = gmsh.model.occ.addPoint(0, Posicion_Traccionado_Y_superior, 0, lc)
p6 = gmsh.model.occ.addPoint(2, Posicion_Traccionado_Y_superior, 0, lc)
p7 = gmsh.model.occ.addPoint(2, Posicion_Traccionado_Y_inferior, 0, lc)
p8 = gmsh.model.occ.addPoint(16, 10, 0, lc)
p9 = gmsh.model.occ.addPoint(18, 0, 0, lc)  # optimizar estos puntos para optimizar
p10 = gmsh.model.occ.addPoint(17, 5, 0, lc)
p11 = gmsh.model.occ.addPoint(19, 5, 0, lc)

# Defino las curvas que conformarán el contorno:
l1 = gmsh.model.occ.add_spline([p2, p10, p3])
l2 = gmsh.model.occ.addLine(p3, p4)
l3 = gmsh.model.occ.addLine(p4, p5)
l4 = gmsh.model.occ.addLine(p5, p6)
l5 = gmsh.model.occ.addLine(p6, p7)
l6 = gmsh.model.occ.addLine(p7, p8)
l7 = gmsh.model.occ.add_spline([p9, p11, p8])
l8 = gmsh.model.occ.addLine(p9, p2)
l9 = gmsh.model.occ.addLine(p3, p8)
l10 = gmsh.model.occ.addLine(p3, p8)
l11 = gmsh.model.occ.addLine(p4, p7)

# Defino los contornos:
C_Traccionado = gmsh.model.occ.addCurveLoop([l3, l4, l5, l11])
C_Pala = gmsh.model.occ.addCurveLoop([l2, l11, l6, l10])
C_Grap = gmsh.model.occ.addCurveLoop([l1, l9, l7, l8])

# Defino las superficies:
S_Traccionado = gmsh.model.occ.addPlaneSurface([C_Traccionado])
S_Pala = gmsh.model.occ.addPlaneSurface([C_Pala])
S_Grap = gmsh.model.occ.addPlaneSurface([C_Grap])
gmsh.model.occ.synchronize()

# Extrudo el Volumen,para eso debo definir el PG y la Entitie de la superficie:1
L_Extrudado = 3
V_Traccionado = gmsh.model.occ.extrude([(2, S_Traccionado)], 0, 0, L_Extrudado)
V_Traccionado_tag = [tag[1] for tag in V_Traccionado if tag[0] == 3][0]
V_Pala = gmsh.model.occ.extrude([(2, S_Pala)], 0, 0, L_Extrudado)
V_Pala_tag = [tag[1] for tag in V_Pala if tag[0] == 3][0]
V_Grap = gmsh.model.occ.extrude([(2, S_Grap)], 0, 0, L_Extrudado)
V_Grap_tag = [tag[1] for tag in V_Grap if tag[0] == 3][0]

gmsh.model.occ.fuse([(3, V_Grap_tag)], [(3, V_Pala_tag), (3, V_Traccionado_tag)], removeObject=False, removeTool=True)

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(dim=3)

# Obtengo nodos, MN, E_Tags, MC
NodeInfo = gmsh.model.mesh.get_nodes()  # saco un objeto con todos los nodos. array uno etiquetas que asigna a los nodos, empieza con 1 y no con 0.
NN = NodeInfo[0].shape[0]  # numero de nodos es la cantidad de elementos uqe tengo en el primer array
NN_Tags = NodeInfo[0] - 1  # Los Tags de los nodos los pongo en numeración python.
MN = NodeInfo[1].reshape(NN, 3)
E_Tags, MC_Flatten = gmsh.model.mesh.get_elements_by_type(
    4)  # dame los tags (numero de elementos) del tipo 4 (tetrahedros) la MC flatten.
NNXE = 4
NE = E_Tags.shape[0]
MC = MC_Flatten.reshape(NE, 4)
MC = MC - np.ones(MC.shape)  # Lo dejo en numeración python

# Busco los tags de la superficie traccionada y del volumen del grap para armar las s.
dl = 1e-3
EntidadTraccionado = gmsh.model.getEntitiesInBoundingBox(-dl, Posicion_Traccionado_Y_inferior - dl, 0 - dl,
                                                         dl, Posicion_Traccionado_Y_superior + dl, L_Extrudado + dl)

EntidadTraccionado = [entidad[1] for entidad in EntidadTraccionado if entidad[0] == 2]
Traccionado_PG = gmsh.model.getPhysicalGroupsForEntity(2, EntidadTraccionado[0])
Tag_V_Grap = [tag[1] for tag in V_Grap if tag[0] == 3]  # me agarro eltag ed dimension 3
Grap_PG = gmsh.model.addPhysicalGroup(3, Tag_V_Grap)
gmsh.model.occ.synchronize()

ETypesTraccionado, ETagsTraccionado, NodeTagsTraccionado = gmsh.model.mesh.getElements(2, EntidadTraccionado[0])
NElementosTraccionados = len(ETagsTraccionado[0])
MC_T = NodeTagsTraccionado[0].reshape(NElementosTraccionados, GLXN) - 1

# Busco los tags del Grap, armo el s y el Us. El grap no se puede mover en ninguna de las 3 direcciones
Tags_Nodos_Grap, CoordFlatten_Nodos_Grap = gmsh.model.mesh.getNodesForPhysicalGroup(3, Grap_PG)
Tags_Nodos_Grap = Tags_Nodos_Grap - 1
s = np.array([Tags_Nodos_Grap * GLXN, Tags_Nodos_Grap * GLXN + 1, Tags_Nodos_Grap * GLXN + 2]).flatten()
Us = np.zeros_like(s)
r = np.array([i for i in range(NN * GLXN) if i not in s]).astype(int)
Fr = np.zeros_like(r).astype(np.float64)
Fx = -97.27
Fy = -90.7

for eT in range(NElementosTraccionados):
    n1 = MC_T[eT, 0].astype(int)
    n2 = MC_T[eT, 1].astype(int)
    n3 = MC_T[eT, 2].astype(int)

    A = (1 / 2) * np.abs(np.linalg.det(np.array([[MN[n1, 1], MN[n1, 2], 1],
                                                 [MN[n2, 1], MN[n2, 2], 1],
                                                 [MN[n3, 1], MN[n3, 2], 1]])))
    Fr[r == n1 * GLXN] += (Fx * A / 3).astype(np.float64)
    Fr[r == n2 * GLXN] += (Fx * A / 3).astype(np.float64)
    Fr[r == n3 * GLXN] += (Fx * A / 3).astype(np.float64)
    Fr[r == n1 * GLXN + 1] += (Fy * A / 3).astype(np.float64)
    Fr[r == n2 * GLXN + 1] += (Fy * A / 3).astype(np.float64)
    Fr[r == n3 * GLXN + 1] += (Fy * A / 3).astype(np.float64)

Kels = []
Ds = []
Bs = []
for e in range(NE):
    Kel, D, B = M3D.k_elemental_3D(MN, MC, nu, E, e)
    Kels.append(Kel)
    Ds.append(D)
    Bs.append(B)

K = M3D.Ensamblado_Matriz_Global(MN, MC, Kels, GLXN)

Fo = np.zeros(len(K))
Fo[r] = Fr
Fo3D = Fo.reshape(NN, GLXN)
Fuerzas_Iniciales = gmsh.view.add('forces')
# por algun motivo le faltaba sumar 1 a nodeinfo
gmsh.view.addModelData(Fuerzas_Iniciales, 0, 'Medio Arco Recurvo', 'NodeData', NodeInfo[0], Fo3D, numComponents=3)
gmsh.option.setNumber(f'View[{Fuerzas_Iniciales}].VectorType', 4)
gmsh.option.setNumber(f'View[{Fuerzas_Iniciales}].GlyphLocation', 2)
gmsh.fltk.run()

U, F = M3D.solve(K, s, r, Us, Fr)

U3D = U.reshape(NN, GLXN)
Desplazamientos = gmsh.view.add("Desplazamientos")
# por algun motivo le faltaba sumar 1 a nodeinfo
deformaciones_model_data = gmsh.view.addModelData(Desplazamientos, 0, 'Medio Arco Recurvo', 'NodeData', NodeInfo[0],
                                                  U3D, numComponents=3)
gmsh.option.setNumber(f'View[{Desplazamientos}].VectorType', 5)

F3D = F.reshape(NN, GLXN)
fuerzas = gmsh.view.add('forces')
# por algun motivo le faltaba sumar 1 a nodeinfo
fuerzas_model_data = gmsh.view.addModelData(fuerzas, 0, 'Medio Arco Recurvo', 'NodeData', NodeInfo[0], F3D,
                                            numComponents=3)
gmsh.option.setNumber(f'View[{fuerzas}].VectorType', 4)
gmsh.option.setNumber(f'View[{fuerzas}].GlyphLocation', 2)

gmsh.fltk.run()
