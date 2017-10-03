#!/usr/bin/env python
# coding=utf-8

from BuildG.buildpjk import *
from Comun.cm import *
from Comun.gn import *
from Comun.hc import *
from Comun.dc import *
from Comun.kc import *
from Comun.dg import *
from Comun.gv import *
import networkx as nx
import sys
import itertools
import random
import math
import itertools
import random
import math
import networkx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from collections import defaultdict

#def main():
def main(argv):
    if len(argv) < 2:
        sys.stderr.write("Usage: %s <input graph>\n" % (argv[0],))
        return 1
    graph_fn = argv[1]
    G = nx.Graph()  #let's create the graph first
    buildG(G, graph_fn, ' ')
    print "Modelo;","Componentes;","Tamano de Grupo maximo;","Nodos;","Enlaces;","densidad;","Diametro;","Radio;","excentricidad;","averagepathlength;","grado promedio;","Cercania;","Centralidad;","Coeficiente de agrupamiento;","Periferia;","Centro;","maximoKcore;","comunidad influyente por grado;","Comunidad influyente por nivel Core"

    '''for i in range(100):
        #G = nx.powerlaw_cluster_graph(1000, 1, 1)
        #G = nx.watts_strogatz_graph(1000, 5, 0.5)
        #G = nx.gnp_random_graph(1000, 1)
        G = nx.barabasi_albert_graph(1000, 5)'''


        #print "Modelo;","Componentes;","Tamano de Grupo maximo;","Nodos;","Enlaces;","densidad;","Diametro;","Radio;","excentricidad;","averagepathlength;","grado promedio;","Cercania;","Centralidad;","Coeficiente de agrupamiento;","Periferia;","Centro;","maximoKcore;","comunidad influyente por grado;","Comunidad influyente por nivel Core"  
    print "metodo louvain"
    modelo = "louvain"
    cm(G,modelo)
    #print "metodo Girvan Newman"
    #modelo = "Girvan"
    #gn(G,modelo)
    #print "metodo Girvan Newman"
    #modelo = "Girvan2"
    #gv(G,modelo)

    #print "cluster jerargico"
    #modelo = "Cluster"
    #hc(G,modelo)
    #print "Dendograma"
    #modelo = "Dendo"
    #dc(G,modelo)
    print "metodo kcore"
    modelo = "kcore"
    kc(G, modelo)
    #print "metodo degree"
    #modelo = "dgree"
    #dg(G, modelo)
    
#main()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
