import networkx as nx
import matplotlib.pyplot as plt
from Comunidad.comunidad import *

def core_number(G):
    if G.is_multigraph():
        raise nx.NetworkXError('MultiGraph and MultiDiGraph types not supported.')

    if G.number_of_selfloops()>0:
        raise nx.NetworkXError('Input graph has self loops; the core number is not defined.','Consider using G.remove_edges_from(G.selfloop_edges()).')

    if G.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors_iter(v),
                                                  G.successors_iter(v)])
    else:
        neighbors=G.neighbors_iter
    degrees=G.degree()
    # sort nodes by degree
    nodes=sorted(degrees,key=degrees.get)
    bin_boundaries=[0]
    curr_degree=0
    for i,v in enumerate(nodes):
        if degrees[v]>curr_degree:
            bin_boundaries.extend([i]*(degrees[v]-curr_degree))
            curr_degree=degrees[v]
    node_pos = dict((v,pos) for pos,v in enumerate(nodes))
    # initial guesses for core is degree
    core=degrees
    nbrs=dict((v,set(neighbors(v))) for v in G)
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos=node_pos[u]
                bin_start=bin_boundaries[core[u]]
                node_pos[u]=bin_start
                node_pos[nodes[bin_start]]=pos
                nodes[bin_start],nodes[pos]=nodes[pos],nodes[bin_start]
                bin_boundaries[core[u]]+=1
                core[u]-=1
    return core



def dg(G,modelo):
       
    prepartition = G.degree() #se halla el diccionario con los grados k de cada nodo
    Edges = G.edges()

    tempgrupo = []
    pregrupo = [] #variable que guardar los grupos de nodos por grado
    for i in range(max(list(prepartition.values()))):
        tempgrupo = []
        for j in prepartition:
            if (i + 1) == prepartition[j]: tempgrupo.append(j)
        if tempgrupo != []: pregrupo.append(tempgrupo)
    #print "partition", prepartition
    #print "values", list(prepartition.values())[0]
    #print "keys", list(prepartition.keys())[0]

    cont = 0

    partition = {}

    for i in pregrupo:
        partition[i[0]] = cont
        for j in i:
            if list(partition.keys()).count(j) == 0: partition[j] = cont
            for k in Edges:
                if k.count(j) > 0: 
                   if k[0] != j:
                      if i.count(k[0]) > 0: 
                         if list(partition.keys()).count(k[0]) == 0: partition[k[0]] = partition[j]
                   if k[1] != j:
                      if i.count(k[1]) > 0: 
                         if list(partition.keys()).count(k[1]) == 0: partition[k[1]] = partition[j]
            cont += 1
    #print partition

    comunidad(G, partition, modelo)

