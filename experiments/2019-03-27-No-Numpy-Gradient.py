from asteval import Interpreter

x = [1., 2., 3., 4., 5., 6.]
y = [1., 4., 9., 16., 25., 36]

def grad(x, y):
    delta_y = [i-j for i,j in zip(y[1:], y)]
    delta_x = [i-j for i,j in zip(x[1:], x)]

    # Fake last element so that length remains the same as inputs.
    return [y/x for x, y in zip(delta_x, delta_y)] + [delta_y[-1]/delta_x[-1]]

grad_func = """
def grad(x, y):
    delta_y = [i-j for i,j in zip(y[1:], y)]
    delta_x = [i-j for i,j in zip(x[1:], x)]

    # Fake last element so that length remains the same as inputs.
    return [y/x for x, y in zip(delta_x, delta_y)] + [delta_y[-1]/delta_x[-1]]
"""
ast = Interpreter()
