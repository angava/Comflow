#!/usr/bin/env python
import networkx as nx
import networkx



def core_number(A):
    if A.is_multigraph():
        raise nx.NetworkXError('MultiGraph and MultiDiGraph types not supported.')

    if A.number_of_selfloops()>0:
        raise nx.NetworkXError('Input graph has self loops; the core number is not defined.','Consider using A.remove_edges_from(A.selfloop_edges()).')

    if A.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([A.predecessors_iter(v),
                                                  A.successors_iter(v)])
    else:
        neighbors=A.neighbors_iter
    degrees=A.degree()
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
    nbrs=dict((v,set(neighbors(v))) for v in A)
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
    
def calc_core_nodos(A,kCore, kGrados,conexion, modelo,tamano): #organiza los nodos para imprimirlos y para hallar los mas influyentes.

    mayor = 0
    kCoreNodos = []
    contnodos = 0
    coreMayor = 0
    kcentral = []
    kcentral = betweenness_centrality(A)
    Cluster =[]
    Cluster = clustering(A)
    Nodos = A.number_of_nodes()
    
    #hallo el nodo con mayor grado.
    maxdgre = max(list(kGrados.values())) #guarda el mayor valor de grado
    temp = []
    for i in kGrados:
        if kGrados[i] == maxdgre: temp.append(i)
    Maxgrados = temp #guardo los nodoscon mayor grado.
  
    #hallo las comunidades inlfuyentes por nivel kcore.
    MaxKcore = max(list(kCore.values())) 
    temp = []
    for i in kCore:
        if kCore[i] == MaxKcore: temp.append(i)
    MaxiCores = temp
    temp = []
    for i in MaxiCores:
        if kGrados[i] == maxdgre: temp.append(i)
    MaxNodoCore = temp

    #Se halla el maxima comunidad en tamano del grafo original.  
    temp = []
    Maxtamano = max(list(tamano.values()))
    cont = 0
    for i in tamano:
        if tamano[i] == Maxtamano: temp.append(i)
        cont += 1
    MaxNodoTamano = temp

    #Se busca el promedio de las medidas del grafo.
    sumaGrados = 0
    sumaCentral = 0
    sumaCluster = 0
    sumaClosenss = 0
    sumaExcent = 0
    if conexion == 1: Excent = eccentricity(A)
    for i in A.nodes(): 
        sumaGrados += kGrados[i]
        sumaCentral += kcentral[i]
        sumaCluster += Cluster[i]
        sumaClosenss += closeness_centrality(A,i)
        if conexion == 1: sumaExcent += Excent[i]
  
    
    Nodos = A.number_of_nodes()

    if conexion == 1: print modelo, ";",cont, ";",MaxNodoTamano, "-", Maxtamano,";", A.number_of_nodes(), ";", A.number_of_edges(),";", density(A),";", max(Excent),";", min(Excent),";", sumaExcent/Nodos,";", nx.average_shortest_path_length(A),";", sumaGrados/Nodos,";", sumaClosenss/Nodos,";", sumaCentral/Nodos,";", nx.average_clustering(A),";", periphery(A),";", center(A),";", MaxKcore,";", Maxgrados, "-", maxdgre,";", MaxNodoCore

    if conexion == 0: print modelo, ";",cont, ";",MaxNodoTamano, "-", Maxtamano,";", A.number_of_nodes(), ";", A.number_of_edges(),";", density(A),";", "0;", "0;", "0;", sumaGrados/Nodos,";", sumaClosenss/Nodos,";", sumaCentral/Nodos,";", nx.average_clustering(A),";", "0;", "0;", MaxKcore,";",  Maxgrados, "-", maxdgre, ";", MaxNodoCore

def closeness_centrality(A, u=None, distance=None, normalized=True):
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight 
        path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                        weight=distance)
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = A.nodes()
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(A,n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(A) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if normalized:
                s = (len(sp)-1.0) / ( len(A) - 1 )
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality

def betweenness_centrality(A, k=None, normalized=True, weight=None,
                           endpoints=False,
                           seed=None):

    betweenness = dict.fromkeys(A, 0.0)  # b[v]=0 for v in A
    if k is None:
        nodes = A
    else:
        random.seed(seed)
        nodes = random.sample(A.nodes(), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma = _single_source_shortest_path_basic(A, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma = _single_source_dijkstra_path_basic(A, s, weight)
        # accumulation
        if endpoints:
            betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness = _accumulate_basic(betweenness, S, P, sigma, s)
    # rescaling
    betweenness = _rescale(betweenness, len(A),
                           normalized=normalized,
                           directed=A.is_directed(),
                           k=k)
    return betweenness

def _single_source_shortest_path_basic(A, s):
    S = []
    P = {}
    for v in A:
        P[v] = []
    sigma = dict.fromkeys(A, 0.0)    # sigma[v]=0 for v in A
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in A[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma


def _single_source_dijkstra_path_basic(A, s, weight='weight'):
    # modified from Eppstein
    S = []
    P = {}
    for v in A:
        P[v] = []
    sigma = dict.fromkeys(A, 0.0)    # sigma[v]=0 for v in A
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in A[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma


def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness

def _rescale(betweenness, n, normalized, directed=False, k=None):
    if normalized is True:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 1.0 / 2.0
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

def clustering(A, nodes=None, weight=None):
    if A.is_directed():
        raise NetworkXError('Clustering algorithms are not defined ',
                            'for directed graphs.')
    if weight is not None:
        td_iter=_weighted_triangles_and_degree_iter(A,nodes,weight)
    else:
        td_iter=_triangles_and_degree_iter(A,nodes)

    clusterc={}

    for v,d,t in td_iter:
        if t==0:
            clusterc[v]=0.0
        else:
            clusterc[v]=t/float(d*(d-1))

    if nodes in A: 
        return list(clusterc.values())[0] # return single value
    return clusterc

def _triangles_and_degree_iter(A,nodes=None):

    if A.is_multigraph():
        raise NetworkXError("Not defined for multigraphs.")

    if nodes is None:
        nodes_nbrs = A.adj.items()
    else:
        nodes_nbrs= ( (n,A[n]) for n in A.nbunch_iter(nodes) )

    for v,v_nbrs in nodes_nbrs:
        vs=set(v_nbrs)-set([v])
        ntriangles=0
        for w in vs:
            ws=set(A[w])-set([w])
            ntriangles+=len(vs.intersection(ws))
        yield (v,len(vs),ntriangles)


def _weighted_triangles_and_degree_iter(A, nodes=None, weight='weight'):
    if A.is_multigraph():
        raise NetworkXError("Not defined for multigraphs.")

    if weight is None or A.edges()==[]:
        max_weight=1.0
    else:
        max_weight=float(max(d.get(weight,1.0) 
                             for u,v,d in A.edges(data=True)))
    if nodes is None:
        nodes_nbrs = A.adj.items()
    else:
        nodes_nbrs= ( (n,A[n]) for n in A.nbunch_iter(nodes) )

    for i,nbrs in nodes_nbrs:
        inbrs=set(nbrs)-set([i])
        weighted_triangles=0.0
        seen=set()
        for j in inbrs:
            wij=A[i][j].get(weight,1.0)/max_weight
            seen.add(j)
            jnbrs=set(A[j])-seen # this keeps from double counting
            for k in inbrs&jnbrs:
                wjk=A[j][k].get(weight,1.0)/max_weight
                wki=A[i][k].get(weight,1.0)/max_weight
                weighted_triangles+=(wij*wjk*wki)**(1.0/3.0)
        yield (i,len(inbrs),weighted_triangles*2)

def eccentricity(A, v=None, sp=None):
    """Return the eccentricity of nodes in A.

    The eccentricity of a node v is the maximum distance from v to
    all other nodes in A.

    Parameters
    ----------
    A : NetworkX graph
       A graph

    v : node, optional
       Return value of specified node       

    sp : dict of dicts, optional       
       All pairs shortest path lengths as a dictionary of dictionaries

    Returns
    -------
    ecc : dictionary
       A dictionary of eccentricity values keyed by node.
    """
#    nodes=
#    nodes=[]
#    if v is None:                # none, use entire graph 
#        nodes=A.nodes()
#    elif v in A:               # is v a single node
#        nodes=[v]
#    else:                      # assume v is a container of nodes
#        nodes=v
    order=A.order()

    e={}
    for n in A.nbunch_iter(v):
        if sp is None:
            length=networkx.single_source_shortest_path_length(A,n)
            L = len(length)
        else:
            try:
                length=sp[n]
                L = len(length)
            except TypeError:
                raise networkx.NetworkXError('Format of "sp" is invalid.')
        if L != order:
            msg = "Graph not connected: infinite path length"
            raise networkx.NetworkXError(msg)
            
        e[n]=max(length.values())

    if v in A:
        return e[v]  # return single value
    else:
        return e

def diameter(A, e=None):
    """Return the diameter of the graph A.

    The diameter is the maximum eccentricity.

    Parameters
    ----------
    A : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    Returns
    -------
    d : integer
       Diameter of graph

    See Also
    --------
    eccentricity
    """
    if e is None:
        e=eccentricity(A)
    return max(e.values())

def periphery(A, e=None):
    """Return the periphery of the graph A. 

    The periphery is the set of nodes with eccentricity equal to the diameter. 

    Parameters
    ----------
    A : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    Returns
    -------
    p : list
       List of nodes in periphery
    """
    if e is None:
        e=eccentricity(A)
    diameter=max(e.values())
    p=[v for v in e if e[v]==diameter]
    return p

def center(A, e=None):
    """Return the center of the graph A. 

    The center is the set of nodes with eccentricity equal to radius. 

    Parameters
    ----------
    A : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    Returns
    -------
    c : list
       List of nodes in center
    """
    if e is None:
        e=eccentricity(A)
    # order the nodes by path length
    radius=min(e.values())
    p=[v for v in e if e[v]==radius]
    return p

def density(A):
    r"""Return the density of a graph.

    The density for undirected graphs is

    .. math::

       d = \frac{2m}{n(n-1)},

    and for directed graphs is

    .. math::

       d = \frac{m}{n(n-1)},

    where `n` is the number of nodes and `m`  is the number of edges in `A`.

    Notes
    -----
    The density is 0 for a graph without edges and 1 for a complete graph.
    The density of multigraphs can be higher than 1.

    Self loops are counted in the total number of edges so graphs with self
    loops can have density higher than 1.
    """
    n=A.number_of_nodes()
    m=A.number_of_edges()
    if m==0 or n <= 1:
        d=0.0
    else:
        if A.is_directed():
            d=m/float(n*(n-1))
        else:
            d= m*2.0/float(n*(n-1))
    return d

def metricas(A, conexion, modelo,tamano):
        
    calc_core_nodos(A,core_number(A), A.degree(),conexion, modelo, tamano)
   
 

