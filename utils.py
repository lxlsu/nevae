import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import tensorflow as tf
import ast
from numpy.linalg import svd, qr, norm
import glob

def slerp(p0, p1, t):
    omega = arccos(dot(p0/norm(p0), p1/norm(p1)))
    so = sin(omega)
    #print "Debug", p0, p1, omega, so,  sin((1.0-t)*omega)/so,  sin((1.0-t)*omega)/so *np.array(p0)
    return sin((1.0-t)*omega) / so * np.array(p0) + sin(t*omega)/so * np.array(p1)

def lerp(p0, p1, t):
    return np.add(p0, t * np.subtract(p1,p0))

def degree(A):
    return np.zeros()


def construct_feed_dict(lr,dropout, k, n, d, decay, placeholders):
    # construct feed dictionary
    feed_dict = dict()


    #feed_dict.update({placeholders['features']: features})
    #feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['lr']: lr})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['decay']: decay})
    #feed_dict.update({placeholders['input']:np.zeros([k,n,d])})
    return feed_dict


def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def basis(adj, atol=1e-13, rtol=0):
    """Estimate the basis of a matrix.


    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    b : ndarray
        The basis of the columnspace of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
    """

    A = degree(adj) - adj

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    q, r = qr(A)
    return q[:rank]

def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))

def get_basis(mat):
    basis = np.zeros(1,1)
    return basis

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_edges(adj):
    G.edges()
    return

def pickle_load(path):
    '''Load the picke data from path'''
    with open(path, 'rb') as f:
        loaded_pickle = pickle.load(f)
    return loaded_pickle

def load_embeddings(fname):
    embd = []
    with open(fname) as f:
        for line in f:
            embd.append(ast.literal_eval(line))
    return embd


def load_data(filename, num=0):
    path = filename
    adjlist = []
    featurelist = []
    edgelist = []
    neg_edgelist = []
    edge_binary = [] 
    posedges = []
    negedges = []
    for fname in sorted(glob.glob(path+"*")):
        f = open(fname, 'r')
        try:
            G=nx.read_edgelist(f, nodetype=int)
        except:
            continue
        f.close()
        n = num
        Gcom = nx.complement(G)
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        degreemat = np.zeros((n,1), dtype=np.float)
        edge_bin = np.zeros((n * (n-1)/2, 2), dtype=np.int)
        edges = G.edges()
        i = 0
        posedge = []
        for (u,v) in edges:
            edge_bin[i][0] = u * n + v
            posedge.append(u*n+v)
            edge_bin[i][1] = 1
            i +=1
        neg_edges = Gcom.edges()
        negedge = []
        for (u,v) in neg_edges:
            if u == v:
                continue
            edge_bin[i][0] = u * n + v
            negedge.append(u*n+v)
            edge_bin[i][1] = -1
            i +=1

        for u in G.nodes():
            degreemat[int(u)][0] = (G.degree(u)*1.0)/(n-1)
        try:
            edgelist.append(G.edges())
            neg_edgelist.append(neg_edges)
            edge_binary.append(edge_bin)
            adjlist.append(np.array(nx.adjacency_matrix(G).todense()))
            featurelist.append(degreemat)
            posedges.append(posedge)
            negedges.append(negedge)
        except:
            continue
    return (adjlist, featurelist, edgelist, neg_edgelist, edge_binary, posedges, negedges)


def pickle_save(content, path):
    '''Save the content on the path'''
    with open(path, 'wb') as f:
        pickle.dump(content, f)
