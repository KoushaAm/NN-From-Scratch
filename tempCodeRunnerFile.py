a = Value(3.0, label = 'a')
b = a + a ; b.label = 'b' # the gradient of a should be 2 d/da(a+a) = 2
b.backward_seq() # a.grad should be 2 not 1 (IT'S A BUG)
draw_dots(b).render(view=True)