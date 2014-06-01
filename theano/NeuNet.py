"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
import numpy
from numpy import float32
import theano
import theano.tensor as T
from LogisticRegr import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.inOutDim=(n_in, n_out)
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
        self.activation = activation

    def activate(self, inputs):
        lin_output = T.dot(inputs, self.W) + self.b
        self.output=lin_output if self.activation is None else self.activation(lin_output)
        return self.output

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, hiddenLayerList, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type hiddenLayerList: [HiddenLayer instances]
        :param hiddenLayerList: A list of hidden layers

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # connect hidden layers (no need to, they're already connected outside when building them)
        self.hiddenLayers=hiddenLayerList
        # prevLy=hiddenLayerList[0]
        # prevLy.input=input
        # for ly in hiddenLayerList[1:]:
        #     ly.input=prevLy.output
        #     prevLy=ly

        # The logistic regression layer gets as input the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=hiddenLayerList[-1].output,
            n_in=hiddenLayerList[-1].inOutDim[1],
            n_out=n_out)

        # symbolic variables for data
        self.X=self.hiddenLayers[0].input # training data
        self.y=T.bvector('y') # labels for training data

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for ly in self.hiddenLayers:
            self.L1 += abs(ly.W).sum()

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
        for ly in self.hiddenLayers:            
            self.L2_sqr += (ly.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative log likelihood of the output
        # of the model, computed in the logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the all layers
        self.params=self.logRegressionLayer.params
        for ly in self.hiddenLayers:
            self.params+=ly.params

    def predictProba(self,inputs):
        prevLy=self.hiddenLayers[0]
        prevLy.activate(inputs)
        for ly in self.hiddenLayers[1:]:
            ly.activate(prevLy.output)
            prevLy=ly
        return self.logRegressionLayer.activate(ly.output)

    def buildTrainFunc(self, trSetX, trSetY, batchSize, default_learningRate=0.1, L1_reg=0, L2_reg=0):
        '''Generates the training function.

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: The training data.

        :type batch_size: int
        :param batch_size: size of a minibatch
        :type default_learningRate: float
        :param default_learningRate: Initial learning rate
        :type L1_regu: float
        :param L1_regu: regularization parameter for L1
        :type L2_regu: float
        :param L2_regu: regularization parameter for L2

        '''
        index = T.lscalar('index')  # index to a [mini]batch
        epoch = T.lscalar('epoch')  # epoch to reduce learning rate
        self.learningRate = theano.shared(float32(default_learningRate), 'lr')# learning rate to use
        # cost function
        cost = self.negative_log_likelihood(self.y)+L1_reg * self.L1+L2_reg * self.L2_sqr
        # gradient of cost with respect to weights in Neu Net
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learningRate * gparam))
        #updates.append((learningRate, T.cast(learningRate/(1+0.001*epoch),dtype=theano.config.floatX)))
        
        trainFunc = theano.function(inputs=[index], outputs=cost,#function(inputs=[index, epoch], outputs=cost,
                    updates=updates,
                    givens={self.X: trSetX[index * batchSize:(index + 1) * batchSize],
                            self.y: trSetY[index * batchSize:(index + 1) * batchSize]})
        return trainFunc