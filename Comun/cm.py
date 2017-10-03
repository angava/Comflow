#!/usr/bin/env python
from Comunidad.comunidad import *
import networkx as nx
import community
import matplotlib.pyplot as plt

def cm(G, modelo):

    partition = community.best_partition(G)    
    comunidad(G, partition, modelo)
        
