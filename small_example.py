import theano
from theano import tensor as T
import numpy as np

def pause():
    raw_input("PRESS ENTER TO CONTINUE.")

def init_weights(shape, amp):
    n_elems = np.product(shape)
    flat_weights = np.array(np.random.randn(n_elems) * amp, dtype=theano.config.floatX)
    s_flat_weights = theano.shared(flat_weights)
    s_weights = s_flat_weights.reshape(shape)
    return s_flat_weights, s_weights

def ReLU(x):
    return T.maximum(x, 0.0)

def model(X, w_h1, w_o):
    h1 = ReLU(T.dot(X, w_h1))
    dot = T.dot(h1, w_o)
    pyx = T.exp(dot) / T.exp(dot).sum(axis=1, keepdims=True)  # softmax
    return pyx

flat_w1, w1 = init_weights((784, 20), 0.01)
flat_wo, wo = init_weights((20, 10), 0.01)
params = [w1, wo]

X = T.fmatrix() # input
Y = T.fmatrix() # output
cost = T.mean(T.nnet.categorical_crossentropy(model(X, w1, wo), Y))
gradient = T.grad(cost=cost, wrt=w1)
hessian = T.hessian(cost=cost, wrt=flat_w1)

hessianFunc = theano.function(inputs=[X, Y], outputs=hessian, allow_input_downcast=True)

print hessianFunc