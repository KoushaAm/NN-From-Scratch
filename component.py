
from graphviz import Digraph


class Value:

    # the value must have set of it's children  from which it was created
    def __init__(self, data, children = (), op = '', label = ''):
        self.data = data
        self.grad = 0.0 # derivate of its output with respoct to itself 
        self.prev = set(children)
        self.op = op
        self.label = label


    def __repr__(self):
        return f'Value(data = {self.data})'
    
    #defin operators
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')
 
    def __sub__(self, other): 
        return Value(self.data - other.data, (self, other), '-')



def lol():
  
  h = 0.001
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L1 = L.data
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'; d.grad = 2.0
  f = Value(-2.0, label='f'); f.grad = 4.0
  f.data += h  # derivative wrt f  gives you the gradient of f
  L = d * f; L.label = 'L'; L.grad = 1.0
  
  L2 = L.data

  
  print((L2 - L1)/h)

  return L

lol()

# Visulize the expression

def trace (root): 

    # build set of nodes and edges 
    nodes, edges = set(), set()

    def build(n):
        if n not in nodes:
            nodes.add(n)
            for child in n.prev: 
                edges.add((child, n))
                # use recursive process to find every children of the value
                build(child)
    
    build(root)
    return nodes, edges

def draw_dots(root):
    
    dot = Digraph(format = 'svg', graph_attr={'rankdir': 'LR'}) 

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n.op, label = n.op)
            # and connect this node to it
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges: 
        #connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot


graph = draw_dots(lol())
graph.render(view=True) 
            
