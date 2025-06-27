import netket as nk
from netket import experimental as nkx
import numpy as np


def ColoredJ1J2(extent, pbc, *, back_diag=True):
    L1, L2 = extent

    def k(i, j):
        return (i % L1) * L2 + (j % L2)

    edges = []
    for i in range(L1):
        for j in range(L2 - 1 + pbc):
            edges.append((k(i, j), k(i, j + 1), 0))
    for i in range(L1 - 1 + pbc):
        for j in range(L2):
            edges.append((k(i + 1, j), k(i, j), 0))
    for i in range(L1 - 1 + pbc):
        for j in range(L2 - 1 + pbc):
            edges.append((k(i, j), k(i + 1, j + 1), 1))
            if back_diag:
                edges.append((k(i + 1, j), k(i, j + 1), 1))

    graph = nk.graph.Graph(edges=edges, n_nodes=L1 * L2)
    return graph

def J1J2OneD(L, J2, pbc, use_marshall=True, total_sz=None):
    # Define custom graph
    edge_colors = []
    for i in range(L):
        # Nearest-neighbor bonds
        if pbc:
            edge_colors.append([i, (i + 1) % L, 1])
        else:
            if i < L - 1:
                edge_colors.append([i, i + 1, 1])
        
        # Next-nearest-neighbor bonds
        if pbc:
            edge_colors.append([i, (i + 2) % L, 2])
        else:
            if i < L - 2:
                edge_colors.append([i, i + 2, 2])

    g = nk.graph.Graph(edges=edge_colors)

    # Sigma^z * Sigma^z interaction operator
    sigmaz = np.array([[1, 0], [0, -1]])
    mszsz = np.kron(sigmaz, sigmaz)

    # Exchange (off-diagonal) operator
    exchange = np.array([[0, 0, 0, 0],
                         [0, 0, 2, 0],
                         [0, 2, 0, 0],
                         [0, 0, 0, 0]])

    # Set rotation factors based on whether Marshall's sign rule is used.
    # When disabled, all factors are 1, so no additional sign is applied.
    if use_marshall:
        rotate_factors = np.array([(-1)**i for i in range(L)])
    else:
        rotate_factors = np.ones(L)

    # Separate edges based on color (bond type)
    edges = g.edges()
    edge_color_list = [color for (_, _, color) in edge_colors]
    nn_edges = [e for e, c in zip(edges, edge_color_list) if c == 1]
    nnn_edges = [e for e, c in zip(edges, edge_color_list) if c == 2]

    # Helper function to compute the product of rotation factors for an edge.
    def compute_gf(edges):
        if edges:
            i, j = edges[0]  # Assumes uniform sign factor per bond type.
            return rotate_factors[i] * rotate_factors[j]
        return 1.0

    gf_nn = compute_gf(nn_edges)
    gf_nnn = compute_gf(nnn_edges)

    # Construct bond operators for both nearest and next-nearest neighbor bonds.
    bond_operator = {
        1: [
            mszsz.tolist(),
            (gf_nn * exchange).tolist()
        ],
        2: [
            (J2 * mszsz).tolist(),
            (J2 * gf_nnn * exchange).tolist()
        ],
    }

    # Flatten the bond operators and assign colors to each.
    flattened_bond_ops = []
    bond_colors = []
    for color, ops in bond_operator.items():
        for op in ops:
            flattened_bond_ops.append(op)
            bond_colors.append(color)

    hilbert = nk.hilbert.Spin(s=0.5, total_sz=total_sz, N=g.n_nodes)
    hamiltonian = nk.operator.GraphOperator(
        hilbert=hilbert,
        graph=g,
        bond_ops=flattened_bond_ops,
        bond_ops_colors=bond_colors,
    ) / 4

    return hamiltonian, g, hilbert

def Hubbard(hilbert, graph, U):
    def c(i, sz):
        return nkx.operator.fermion.destroy(hilbert, i, sz)

    def c_dag(i, sz):
        return nkx.operator.fermion.create(hilbert, i, sz)

    def n(i, sz):
        return nkx.operator.fermion.number(hilbert, i, sz)

    up = +1
    down = -1
    H = 0
    for i, j in graph.edges():
        for sz in [up, down]:
            H -= c_dag(i, sz) * c(j, sz) + c_dag(j, sz) * c(i, sz)
    for i in graph.nodes():
        H += U * n(i, up) * n(i, down)
    return H


def tVModel(hilbert, graph, V):
    def c(i):
        return nkx.operator.fermion.destroy(hilbert, i)

    def c_dag(i):
        return nkx.operator.fermion.create(hilbert, i)

    def n(i):
        return nkx.operator.fermion.number(hilbert, i)

    H = 0
    for i, j in graph.edges():
        H -= c_dag(i) * c(j) + c_dag(j) * c(i)
        H += V * n(i) * n(j)
    return H
