class Graph:
  def __init__(self):
    self.blocks = []
  
  def add_block(self, block):
    self.blocks.append(block)

  def vis(self):
    for i, block in enumerate(self.blocks):
        pass

import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()
G.add_node(1,pos=(1,1), name='haha')
G.add_node(2,pos=(1,2))
G.add_node('haha',pos=(2,2))
G.add_edge(1,2)
G.add_edge(1,3)
pos=nx.get_node_attributes(G,'pos')
plt.figure(figsize=(8,8))
nx.draw(G,pos, with_labels=True)
plt.savefig('./test.png')