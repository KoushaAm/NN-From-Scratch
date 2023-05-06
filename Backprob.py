
from graphviz import Digraph
import numpy as np
from numpy import tanh
import math
class Value:

    # the value must have set of it's children  from which it was created
    def __init__(self, data, children = (), op = '', label = ''):
        self.data = data
        self.grad = 0.0 # derivate of its output with respoct to itself 
        self.prev = set(children)
        self.op = op
        self.label = label
        # the backward fn requires mul by grad becuase of chain rule
        self.backward = lambda: None # by default, a function to do nothing
        
    def __repr__(self):
        return f'Value(data = {self.data})'
    
    #defin operators
    def __add__(self, other):
        res = Value(self.data + other.data, (self, other), '+')

        def backward() : # L = b + 1 --> dL/db = 1
            self.grad += 1.0 * res.grad # local derivative * the derivative of last node
            other.grad += 1.0 * res.grad

        res.backward = backward

        return res
    
    def __mul__(self, other):
        res = Value(self.data * other.data, (self, other), '*')

        def backward() : # L = b * 2--> dL/db = 1 * the derivate wrt next node  chain rule 
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad

        res.backward = backward

        return res
    
    
 
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        res = Value(t, (self,), 'tanh')

        def backward():
            self.grad += (1 - t**2) * res.grad
        
        res.backward = backward
        return res
    
    def backward_seq(self): 
        #utalizes topological ordering to do backprop 
        # by calling backward on each node in the graph
        self.grad = 1.0
        topo = []
        print("backward called")
        visited = set()

        def sort_topo(node):
            if node not in visited: 
                visited.add(node)
                for child in node.prev: 
                    sort_topo(child)
                
                topo.append(node)
            
            
        sort_topo(self)
        print(topo)

        for node in reversed(topo):
            node.backward()
        
                

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


# dd/dc = 1.0
# dd/de = 1.0
# how do values impact each other its children
# how does c impact L?
# dL/dc = dL/dd * dd/dc = f= (-2) * 1 = -4


# we can tune the output L 
# by a.data += (0.01 * a.grad),  b.data += (0.01 * b.grad),...
# this way L becomes less negative (as layers go forward)
 

#make an activation function 
# tanh(x) = (e^x - e^-x)/(e^x + e^-x)

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'




# topolocial ordering in backpropagation
o.backward_seq()
#show the graph
# graph = draw_dots(o)
# graph.render(view=True)


a = Value(3.0, label = 'a')
b = a + a ; b.label = 'b' # the gradient of a should be 2 d/da(a+a) = 2
b.backward_seq() # a.grad should be 2 not 1 (IT'S A BUG)
draw_dots(b).render(view=True)

# in order to resolve the issue with repeated varible we must accumulate the gradients 
# by += their previous gradient for a new operator
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b    ; d.label = 'd'
e = a + b    ; e.label = 'e'
f = d * e    ; f.label = 'f'

f.backward_seq()
draw_dots(f).render(view=True)





#************ JUNK ************

# def lol():
  
#   h = 0.001
  
#   a = Value(2.0, label='a')
#   b = Value(-3.0, label='b')
#   c = Value(10.0, label='c')
#   e = a*b; e.label = 'e'
#   d = e + c; d.label = 'd'
#   f = Value(-2.0, label='f')
#   L = d * f; L.label = 'L'
#   L1 = L.data
  
#   a = Value(2.0, label='a')
#   b = Value(-3.0, label='b')
#   b.data += h
#   c = Value(10.0, label='c')
#   e = a*b; e.label = 'e'
#   d = e + c; d.label = 'd'
#   f = Value(-2.0, label='f')
#   L = d * f; L.label = 'L'
#   L2 = L.data
  
#   print((L2 - L1)/h)

#   return L

# lol()


# since out.grad is set to 0 by default we should set o.grad to 1 as basecase
# o.grad = 1.0
# o.backward()
# n.backward()
# b.backward() # a leaf node, by initilization backward : labmda None
# x1w1x2w2.backward()
# x1w1.backward()
# x2w2.backward()