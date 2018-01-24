import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import tensorflow as tf
import ast
import scipy
from numpy.linalg import svd, qr, norm
import glob

def normalise_weighted(prob, weight, n, bin_dim, seen_list, list_edges, indicator):

    prob = np.reshape(prob, [n, n])
    
    #prob = np.multiply(np.reshape(prob, [n, n]), indicator)
    print "Debug: Prob", prob
    weight = np.reshape(weight, [n, n, bin_dim])
    weight = np.multiply(weight, indicator)
    problist = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j, 1) in seen_list or (i, j, 2) in seen_list or (i, j, 3) in seen_list:
                if (i, j, 1) in list_edges:
                    list_edges.remove((i, j, 1))
                if (i, j, 2) in list_edges:
                    list_edges.remove((i, j, 2))
                if (i, j, 3) in list_edges:
                    list_edges.remove((i, j, 3))
                continue

            problist.extend(prob[i][j] * weight[i][j])
    p = np.array(problist)
    p /= p.sum()

    return list_edges, p

def normalise(prob, weight, n, bin_dim):
    p_rs = prob
    #p_rs = prob/prob.sum(axis=0)[:,None] 
    p_new_rs = np.zeros([n,n,bin_dim])
    w_rs = np.zeros([n, n, bin_dim])
    problist = []
    negval = 0.0
    for i in range(n):
        for j in range(i+1, n):
            problist.append(prob[i][j] * (1 - (weight[i][j][0]/sum(weight[i][j]))))
            #print len(weight[i][j])
            '''
            w_rs[i][j] = weight[i][j]
            #/ sum(weight[i][j])
            #p_new_rs[i][j] = p_rs[i][j] * w_rs[i][j]
            problist.extend(p_rs[i][j]* w_rs[i][j])
            negval += p_rs[i][j] * w_rs[i][j][0]
            '''
    #print len(problist), negval
    #prob = np.triu(p_new_rs,1)
    #problist.append(negval)
    problist = np.array(problist)
    #print problist.sum()
    return problist/problist.sum()


def slerp(p0, p1, t):
    omega = arccos(dot(p0/norm(p0), p1/norm(p1)))
    so = sin(omega)
    #print "Debug", p0, p1, omega, so,  sin((1.0-t)*omega)/so,  sin((1.0-t)*omega)/so *np.array(p0)
    return sin((1.0-t)*omega) / so * np.array(p0) + sin(t*omega)/so * np.array(p1)

def lerp(p0, p1, t):
    return np.add(p0, t * np.subtract(p1,p0))

def degree(A):
    n = A[0].shape[0]
    degree = np.zeros((n,n))
    for i in range(n):
        degree[i][i] = np.sum(A[i])
    return degree

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
    return q

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


def load_data(filename, num=0, bin_dim=3):
    path = filename+"/*"
    adjlist = []
    featurelist = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    #for findex in range(40):
    #filenumber = 1
    filenumber = int(len(glob.glob(path)) * 0.9)
    for fname in sorted(glob.glob(path))[:filenumber]:
        #fname = filename+"/"+str(findex+1)+".edgelist"
        #fname = filename+"/"+str(findex+1)+".txt"
        f = open(fname, 'r')
        lines = f.read().strip()
        linesnew = lines.replace('{', '{\'weight\':').split('\n')
        try:
            #G=nx.read_edgelist(f, nodetype=int)
            G=nx.parse_edgelist(linesnew, nodetype=int)
        except:
            print "Except"
            continue
        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        degreemat = np.zeros((n,1), dtype=np.float)

        totaledgelist = list(G.edges())
        maxel = -1
        maxdeg = -1

        for u in G.nodes():
            #degreemat[int(u)][0] = int(G.degree(u)) * 2.0 / n
            degreemat[int(u)][0] = (G.degree(u)*1.0)/(n-1)
            if G.degree(u) > maxdeg :
                    maxdeg = G.degree(u)
                    maxel = u

        try:
            weight = np.array(nx.adjacency_matrix(G).todense())
            adj = np.zeros([n,n])
            weight_bin = np.zeros([n,n,bin_dim])
            for i in range(n):
                for j in range(n):
                    if weight[i][j]>0:
                        adj[i][j] = 1
                    weight_bin[i][j][weight[i][j]] = 1
            #adj = scipy.sign(x)    
            adjlist.append(adj)
            bfs_edgelist = list(nx.bfs_edges(G, maxel)) 
            for i in range(len(bfs_edgelist)):
                (u, v) = bfs_edgelist[i] 
                if int(u) > int(v):
                    bfs_edgelist[i] = (v, u)
            diff_edges = list(set(totaledgelist) - set(bfs_edgelist))
            temp_list = []
            for (u,v) in (bfs_edgelist + diff_edges):
                
                temp_list.append((u, v, G[u][v]['weight']))
            edgelist.append(temp_list)
            weightlist.append(weight)
            weight_binlist.append(weight_bin)
            featurelist.append(degreemat)
        except:
            continue
        #print fname
    return (adjlist, weightlist, weight_binlist, featurelist, edgelist)
    #return (nx.adjacency_matrix(G).todense(), degreemat, edges, non_edges)

def calculate_feature(weight, bin_dim):
        n = len(weight[0])
        degreemat = np.zeros((n, 1), dtype=np.float)
        #degreeindicator = np.ones((n,n), dtype=np.float)
        degreeindicator = np.ones((n, n, bin_dim), dtype=np.float)
        adj = np.zeros([n,n])
        weight_bin = np.zeros([n,n,bin_dim])
        #weight_bin = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if weight[i][j]>0:
                    adj[i][j] = 1
                    weight_bin[i][j][int(weight[i][j])-1] = 1
        for u in range(n):
            degreemat[int(u)][0] = np.sum(adj[u])//n
            for v in range(n):
                degv = np.sum(adj[u])
                degu = np.sum(adj[v])
                if degv >= 5 or degu >= 5:
                    degreeindicator[u][v][0] = 0
                if degv >=4 or degu >=4:
                    degreeindicator[u][v][1] = 0
                if degv >=3 or degu >=3:
                    degreeindicator[u][v][2] = 0 
                
                degreeindicator[v][u] = degreeindicator[u][v]
        
        return degreemat, weight_bin, adj, degreeindicator

def proxy(filename, perm = False):
        print "filename", filename
        f = open(filename, 'r')
        G=nx.read_edgelist(f, nodetype=int)
        n = G.number_of_nodes()
        edges = G.edges()
        if perm == True:
            p = np.identity(n, dtype=np.int)
            np.random.shuffle(p)
            adj = np.array(nx.adjacency_matrix(G).todense())
            adj = np.matmul(np.matmul(p,adj),p.transpose())
            return adj
        return np.array(nx.adjacency_matrix(G).todense())

def pickle_save(content, path):
    '''Save the content on the path'''
    with open(path, 'wb') as f:
        pickle.dump(content, f)
