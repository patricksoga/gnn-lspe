import dgl
import torch
import scipy.sparse as sp
import scipy
import random

def multiple_automaton_encodings(g: dgl.DGLGraph, transition_matrix, initial_vector, diag=False, matrix='A', idx=0):
    pe = automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, ret_pe=True, idx=idx)
    key = f'pos_enc_{idx}'
    # if 'pos_enc' not in g.ndata:
    g.ndata[key] = pe
    return g

def add_multiple_automaton_encodings(dataset, transition_matrices, initial_vectors, diag=False, matrix='A'):
    for i, (transition_matrix, initial_vector) in enumerate(zip(transition_matrices, initial_vectors)):
        dataset.train.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.train.graph_lists]
        dataset.val.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.val.graph_lists]
        dataset.test.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.test.graph_lists]

    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset

def automaton_encoding(g, transition_matrix, initial_vector, diag=False, matrix='A', ret_pe=False, idx=0):
    """
    Graph positional encoding w/ automaton weights
    """
    # transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal

    # if diag:
    #     transition_matrix = torch.diag(transition_matrix)
        # torch.einsum('ij, kj -> ij', a, b) for matrix
        # torch.einsum('ij, j->ij', a, b) for vector
        # torch.einsum('ij, i->ij', a, b), a is a matrix, b is a vector

    if diag:
        transition_inv = transition_matrix**-1
    else:
        transition_inv = torch.inverse(transition_matrix).cpu().numpy()

    if matrix == 'A':
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
    elif matrix == 'L':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - N * A * N
        mat = L.todense()
    elif matrix == 'SL':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) + N * A * N
        mat = L.todense()
    elif matrix == 'UL':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A - N).todense()
    elif matrix == 'AD':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A + N).todense()

    if idx == 0:
        initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    else:
        pi = torch.zeros(initial_vector.shape[0], g.number_of_nodes())
        index = random.randint(0, g.number_of_nodes()-1)
        pi[:, index] = initial_vector.squeeze(1)
        initial_vector = pi

    if diag:
        mat_product = torch.einsum('ij, i->ij', initial_vector, transition_inv).cpu().numpy()
        transition_inv = torch.diag(transition_inv).cpu().numpy()
    else:
        initial_vector = initial_vector.cpu().numpy()
        mat_product = transition_inv @ initial_vector

    pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
    pe = torch.from_numpy(pe.T).float()

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector, diag=False, matrix='A'):
    # Graph positional encoding w/ pre-computed automaton encoding
    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.test.graph_lists]
    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset