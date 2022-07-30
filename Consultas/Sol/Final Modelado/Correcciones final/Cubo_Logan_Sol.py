import gmsh
import numpy as np
import FuncionesMEF3D as M3D

gmsh.initialize()
gmsh.model.add('Cubo Prueba')

L = 1  # m
lc = L
nu = 0.3
E = 210e9  # Pa

p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(L, L, 0, lc)
p4 = gmsh.model.geo.addPoint(0, L, 0, lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

C = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

S = gmsh.model.geo.addPlaneSurface([C])

V_cubo = gmsh.model.geo.extrude([(2, S)], 0, 0, L)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(dim=3)
gmsh.model.mesh.refine()
gmsh.model.geo.synchronize()

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

GLXN = 3

Kels = []
Ds = []
Bs = []

for e in range(NE):
    Kel, D, B = M3D.k_elemental_3D(MN, MC, nu, E, e)
    Kels.append(Kel)
    Ds.append(D)
    Bs.append(B)

K = M3D.Ensamblado_Matriz_Global(MN, MC, Kels, GLXN)

dl = L * 1e-6
EntidadTraccionado = gmsh.model.getEntitiesInBoundingBox(0, 0, L - dl, L, L, L + dl)
EntidadTraccionado = [entidad[1] for entidad in EntidadTraccionado if entidad[0]==2]
Traccionado_PG = gmsh.model.getPhysicalGroupsForEntity(2, EntidadTraccionado[0])

Empotrado_PG = gmsh.model.addPhysicalGroup(2, [S])
# Traccionado_PG = gmsh.model.addPhysicalGroup(2, [26])

NodosEmpotrado = gmsh.model.mesh.getNodesForPhysicalGroup(2, Empotrado_PG)[0]
# EntidadTraccionado = gmsh.model.getEntitiesForPhysicalGroup(2, Traccionado_PG)
ETypesTraccionado, ETagsTraccionado, NodeTagsTraccionado = gmsh.model.mesh.getElements(2, EntidadTraccionado[0])
NElementosTraccionados = len(ETagsTraccionado[0])
MC_T = NodeTagsTraccionado[0].reshape(NElementosTraccionados, GLXN) - 1

s = (NodosEmpotrado - 1) * GLXN + 2
Us = np.zeros_like(s)
r = np.array([i for i in range(NN * GLXN) if i not in s])
Fr = np.zeros_like(r).astype(float)
Tension = 100

for eT in range(NElementosTraccionados):
    n1 = MC_T[eT, 0].astype(int)
    n2 = MC_T[eT, 1].astype(int)
    n3 = MC_T[eT, 2].astype(int)
    # Aca saque en A el np.abs()
    A = (1 / 2) * np.linalg.det(np.array([[MN[n1, 0], MN[n1, 1], 1],
                                          [MN[n2, 0], MN[n2, 1], 1],
                                          [MN[n3, 0], MN[n3, 1], 1]]))
    # print('nodos: ', n1 * GLXN + 2, n2 * GLXN + 2, n3 * GLXN + 2)
    Fr[r == n1 * GLXN + 2] += Tension * A / 3
    Fr[r == n2 * GLXN + 2] += Tension * A / 3
    Fr[r == n3 * GLXN + 2] += Tension * A / 3

U, F = M3D.solve(K, s, r, Us, Fr)
print('Sum F', F.sum())

U3D = U.reshape(NN, GLXN)

strains = gmsh.view.add("Desplazamientos")
strain_model_data = gmsh.view.addModelData(strains, 0, 'Cubo Prueba', 'NodeData', NodeInfo[0], U3D, numComponents=3)
gmsh.option.setNumber(f'View[{strains}].VectorType', 5)

F3D = F.reshape(NN, GLXN)

Fuerza = gmsh.view.add('forces')
Fuerza_model_data = gmsh.view.addModelData(Fuerza, 0, 'Cubo Prueba', 'NodeData', NodeInfo[0], F3D, numComponents=3)
gmsh.option.setNumber(f'View[{Fuerza}].VectorType', 4)
gmsh.option.setNumber(f'View[{Fuerza}].GlyphLocation', 2)

gmsh.fltk.run()
