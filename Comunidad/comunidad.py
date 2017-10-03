from Comunidad.cmpjk import *
import networkx as nx
import community
import matplotlib.pyplot as plt

def comunidad(G,partition, modelo, hc = None):
   Grupo = []
   tamano = {}
   #first compute the best partition

   #partition = community.best_partition(G)
   #values =  [partition.get(node) for node in G.nodes()]
   #nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=60, with_labels=False)

   #drawing
   size = float(len(set(partition.values())))
   for i in range(int(size)): Grupo.append("")
   pos = nx.spring_layout(G)
   count = 0.
   comp = 0
   for com in set(partition.values()):
       comp += 1
       count = count + 1.
       list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
       if comp/size > 0 and comp/size <= 0.05: colr = "white"
       if comp/size > 0.05 and comp/size <= 0.1:colr = "black", 
       if comp/size > 0.1 and comp/size <= 0.15:colr = "red"
       if comp/size > 0.15 and comp/size <= 0.2:colr = "blue"
       if comp/size > 0.2 and comp/size <= 0.25:colr = "orange"
       if comp/size > 0.25 and comp/size <= 0.3:colr = "green"
       if comp/size > 0.3 and comp/size <= 0.35:colr = "gray"
       if comp/size > 0.35 and comp/size <= 0.4:colr = "brown"
       if comp/size > 0.4 and comp/size <= 0.45:colr = "yellow"
       if comp/size > 0.45 and comp/size <= 0.5: colr = "cyan"
       if comp/size > 0.5 and comp/size <= 0.55: colr = "pink"
       if comp/size > 0.55 and comp/size <= 0.6: colr = "purple"
       if comp/size > 0.6 and comp/size <= 0.65:colr = "violet"
       if comp/size > 0.65 and comp/size <= 0.7:colr = "gold"
       if comp/size > 0.7 and comp/size <= 0.75:colr = "indigo"
       if comp/size > 0.75 and comp/size <= 0.8: colr = "lime"
       if comp/size > 0.8 and comp/size <= 0.85: colr = "silver"
       if comp/size > 0.85 and comp/size <= 0.9: colr = "tan"
       if comp/size > 0.9 and comp/size <= 0.95: colr = "fuchsia"
       if comp/size > 0.95 and comp/size <= 1: colr = "olive"
       nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 60, node_color = colr, with_labels=True)
       print ("Grupo:", comp, "tamano:", len(list_nodes) ,"nodos:", list_nodes)
       tamano[comp] = len(list_nodes)
       Grupo[comp-1] = list_nodes
#   print("numero de componentes", comp)
   #mod = community.modularity(partition,G)
   #print("modularidad:", mod)
   #print(Nodospajek)
   nx.draw_networkx_edges(G,pos, alpha=0.5)
   plt.show()
   #print tamano
   pajek_group(G,Grupo, modelo, tamano, hc)
