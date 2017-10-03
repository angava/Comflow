import networkx as nx
from Metricas.medidas import *
import matplotlib.pyplot as plt

def pajek_group(G, Grupos, modelo, tamano, hc = None):
#    print("comienza reduccion de grafo")

    #elimino nodos con grado cero.

    #nuevo experimento, miraremos si es igual al arrojado por nuestro algoritmo.

    if hc != 1: B = nx.blockmodel(G,Grupos)
    if hc == 1: B = G
    #Re nombro los nodos para eliminar el nodo cero y que todo corresponda a los grupos encontrados en la funcion anterior.
    mapping = {} #variable que guardara el nuevo valor de nodos.
    cont = 0
    for i in B.nodes(): #creo diccionario que guarda el nuevo valor de nodos.
        mapping[cont] = cont + 1
        cont += 1

    A=nx.relabel_nodes(B,mapping) #asigno grafo con el nuevo valor de nodos.

   
    NodosBorrar = []
    cont_group = 1
    cont = 1
    for i in A.nodes():
        if A.degree(cont_group) != 0: 
           cont += 1
        if A.degree(cont_group) == 0: 
           NodosBorrar.append(cont_group)
        cont_group += 1
    

    for i in NodosBorrar: A.remove_node(i)
    Nodes = A.number_of_nodes()
    
#    print "*vertices ", A.number_of_nodes()
    cont = 1
#    for i in A.nodes():
#		print cont,'"Grupo', i, '"'
#		cont += 1 
#    print "*Arcs"
#    for i in A.edges(): print i[0], i[1], int(A[i[0]][i[1]]['weight'])

    conexion = 0
    if A.nodes() == []:print modelo,";0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0" 
    if A.nodes() != []:
       if nx.is_connected(A): conexion = 1
       metricas(A, conexion, modelo, tamano)
 
    pos = nx.spring_layout(A)
    labels={}
    cont = 1 
    for i in A.nodes():
        colr = float(cont)/Nodes
        nx.draw_networkx_nodes(A, pos, [i] , node_size = 250, node_color = str(colr), with_labels=True)
        labels[i] = i
        cont += 1

    nx.draw_networkx_labels(A,pos,labels,font_size=5)        
    nx.draw_networkx_edges(A,pos, alpha=0.5)
    plt.show()
