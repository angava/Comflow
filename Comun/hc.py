#!/usr/bin/env python
# encoding: utf-8
"""
Example of creating a block model using the blockmodel function in NX.  Data used is the Hartford, CT drug users network:

@article{,
    title = {Social Networks of Drug Users in {High-Risk} Sites: Finding the Connections},
    volume = {6},
    shorttitle = {Social Networks of Drug Users in {High-Risk} Sites},
    url = {http://dx.doi.org/10.1023/A:1015457400897},
    doi = {10.1023/A:1015457400897},
    number = {2},
    journal = {{AIDS} and Behavior},
    author = {Margaret R. Weeks and Scott Clair and Stephen P. Borgatti and Kim Radda and Jean J. Schensul},
    month = jun,
    year = {2002},
    pages = {193--206}
}

"""

__author__ = """\n""".join(['Drew Conway <drew.conway@nyu.edu>',
                            'Aric Hagberg <hagberg@lanl.gov>'])

from collections import defaultdict
import networkx as nx
import numpy
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from Comunidad.comunidad import *

def create_hc(G):
   
    """Creates hierarchical cluster of graph G from distance matrix"""
    path_length=nx.all_pairs_shortest_path_length(G)
    distances=numpy.zeros((len(G),len(G)))
    for u,p in path_length.items():
        for v,d in p.items():
            distances[u][v]=d
    # Create hierarchical cluster
    Y=distance.squareform(distances)
    Z=hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrive purposes
    membership=list(hierarchy.fcluster(Z,t=1.15))
    # Create collection of lists for blockmodel
    partition=defaultdict(list)
    for n,p in zip(list(range(len(G))),membership):
        partition[p].append(n)
    return list(partition.values())

def hc(G, modelo):
    
    #G=nx.read_edgelist("test.net")
    # Extract largest connected component into graph H
    components = [comp for comp in nx.connected_component_subgraphs(G)]
    component_size = [len(comp)for comp in components]

    H=components[0]
    # Makes life easier to have consecutively labeled integer nodes
    H=nx.convert_node_labels_to_integers(H)
    # Create parititions with hierarchical clustering
    partitions=create_hc(H)

    partition = {}
    #Convertimos. partitions a partition como diccionario.
    cont = 0
    for i in partitions:
        for j in i:
            partition[j + 1] = cont # se le suma uno para no tener nodo cero sino desde nodo 1.
        cont += 1


    # Build blockmodel graph
    BM=nx.blockmodel(H,partitions)
    hc = 1
    comunidad(BM, partition, modelo, hc)
    hc = 0
