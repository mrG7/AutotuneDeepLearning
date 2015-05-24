import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 400
feats = 784
D = (rng.randn(N,feats), rng.randint(size=N, low = 0, high = 2))

training_steps = 10000

x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(784), name = "w")
b = theano.shared(0., name = "b")

# print "Inital model:"
# print w.get_value(), b.get_value()

p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b))
prediction = p_1 > 0.5

xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01*(w**2).sum()

gw, gb = T.grad(cost, [w,b])

train = theano.function(
                inputs = [x,y],
                outputs=[prediction,xent],
                updates={
                    w : w-0.1*gw,
                    b : b-0.1*gb
                },
                allow_input_downcast=True)

predict = theano.function(inputs = [x], outputs=prediction)

for i in xrange(training_steps):
    print i
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D: ", D[1]
print "predictions on D: ", predict(D[0])