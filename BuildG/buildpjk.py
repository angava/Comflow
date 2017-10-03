#!/usr/bin/env python
import networkx as nx
import math
import csv
import sys


partition = {} #diccionario que almacena los grupos y que luego es usado por la funcion comunidad para hacer su objetivo.

#this method just reads the graph structure from the file
def buildG(G, file_, delimiter_):
    global Nodospajek
    Nodospajek = []
    #construct the weighted version of the contact graph from cgraph.dat file
    #reader = csv.reader(open("/home/kazem/Data/UCI/karate.txt"), delimiter=" ")
    reader = csv.reader(open(file_), delimiter=delimiter_)
    Arcos = 0
    cont = 0
    for line in reader:
        if Arcos == 0 and  line[0] != "*Arcs" and cont != 0:
            Nodospajek.append(line)
        if Arcos == 1:
           if len(line) >  2:
              if float(line[2]) != 0.0:
                #line format: u,v,w
                G.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
           else:
            #line format: u,v
               G.add_edge(int(line[0]),int(line[1]),weight=1.0)
        if line[0] == "*Arcs" : Arcos = 1
        cont = 1




