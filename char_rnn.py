import numpy as np
import pickle

class char_rnn(object):
    """docstring for char_rnn"""
    def __init__(self, txt = ''):
        super(char_rnn, self).__init__()
        if not txt:
            print 'init to generator, no for training.'
            return
        # Training data. should be simple plain text file
        self.data = open(txt, 'r').read()
        char_set = list(set(self.data))
        self.data_size, self.charset_size = len(self.data), len(char_set)
        # keep in mind, eg: a is [1,0,0,0] and b is [0,1,0,0]
        self.char_to_idx = { ch:i for i,ch in enumerate(char_set) }
        self.idx_to_char = { i:ch for i,ch in enumerate(char_set) }

        # temporary parameters
        self.input_state = {}   # x
        self.hidden_state = {}  # h
        self.output_state = {}  # y
        self.prob_state = {}    # probabilities

        # hyper parameters
        self.hidden_size = 100   # size of hidden layer of neurons
        self.seq_length = 25     # number of steps to unroll the RNN for
        self.learning_rate = 1e-1# for Adagrad update

        # model parameters. waiting for learning.(w is mean weight)
        self.w_i2h = np.random.randn(self.hidden_size, self.charset_size)*0.01 # input to hidden
        self.w_h2h = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
        self.w_h2o = np.random.randn(self.charset_size, self.hidden_size)*0.01 # hidden to output
        self.b_h = np.zeros((self.hidden_size, 1)) # hidden bias
        self.b_o = np.zeros((self.charset_size, 1)) # output bias

    def forward(self, inputs, targets, hprev):
        '''
        inputs: list of integers. the idx in char set.
        targets: list of integets.
        hprev: Hx1 array of initial hidden state
        '''
        loss = 0
        self.hidden_state[-1] = np.copy(hprev)
        for t in xrange(self.seq_length):
            # encode in 1-of-k representation
            # eg: a is [1,0,0,0] and b is [0,1,0,0]
            self.input_state[t] = np.zeros((self.charset_size, 1))
            self.input_state[t][inputs[t]] = 1
            # update hidden state
            self.hidden_state[t] = np.tanh(np.dot(self.w_i2h, self.input_state[t]) + \
                np.dot(self.w_h2h, self.hidden_state[t-1]) + self.b_h)
            # compute the output vector
            # unnormalized log probabilities for next chars
            self.output_state[t] = np.dot(self.w_h2o, self.hidden_state[t]) + self.b_o
            # probabilities for next chars
            self.prob_state[t] = np.exp(self.output_state[t]) / np.sum(np.exp(self.output_state[t]))
            # softmax (cross-entropy loss)
            loss += -np.log(self.prob_state[t][targets[t], 0])
            # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            self.prob_state[t][targets[t]] -= 1
        return loss

    def backward(self):
        '''
        backward pass: compute gradients going backwards
        '''
        dw_i2h, dw_h2h, dw_h2o = np.zeros_like(self.w_i2h), np.zeros_like(self.w_h2h), np.zeros_like(self.w_h2o)
        db_h, db_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        dh_next = np.zeros_like(self.hidden_state[0])
        for t in reversed(xrange(self.seq_length)):
            do = np.copy(self.prob_state[t])
            dw_h2o += np.dot(do, self.hidden_state[t].T)
            db_o += do
            dh = np.dot(self.w_h2o.T, do) + dh_next # backprop into h
            dhraw = (1 - self.hidden_state[t] * self.hidden_state[t]) * dh # backprop through tanh nonlinearity
            db_h += dhraw
            dw_i2h += np.dot(dhraw, self.input_state[t].T)
            dw_h2h += np.dot(dhraw, self.hidden_state[t-1].T)
            dh_next = np.dot(self.w_h2h.T, dhraw)
        for dparam in [dw_i2h, dw_h2h, dw_h2o, db_h, db_o]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dw_i2h, dw_h2h, dw_h2o, db_h, db_o, self.hidden_state[self.seq_length-1]

    def training(self):
        # iteration counter and data pointer
        _iter, _p = 0, 0
        # m is memory
        mw_h2i, mw_h2h, mw_h2o = np.zeros_like(self.w_i2h), np.zeros_like(self.w_h2h), np.zeros_like(self.w_h2o)
        # memory bias variables for Adagrad
        mb_h, mb_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        # loss at iteration 0
        smooth_loss = -np.log(1.0/self.charset_size)*self.seq_length

        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if _p+self.seq_length+1 >= len(self.data) or _iter == 0:
                hprev = np.zeros((self.hidden_size, 1)) # reset RNN memory
                _p = 0 # go from start of data
            inputs = [self.char_to_idx[ch] for ch in self.data[_p:_p+self.seq_length]]
            targets = [self.char_to_idx[ch] for ch in self.data[_p+1:_p+self.seq_length+1]]

            # forward seq_length characters through the net
            loss = self.forward(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if _iter % 100 == 0:
                print 'iter %d, loss: %f' % (_iter, smooth_loss) # print progress

            # backward fetch gradient
            dw_i2h, dw_h2h, dw_h2o, db_h, db_o, hprev = self.backward()

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o], 
                                            [dw_i2h, dw_h2h, dw_h2o, db_h, db_o], 
                                            [mw_h2i, mw_h2h, mw_h2o, mb_h, mb_o]):
                mem += dparam * dparam
                # adagrad update
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            _p += self.seq_length # move data pointer
            _iter += 1            # iteration counter

            # output sample txt
            if _iter % 1000 == 0:
                self.generator('a', 200)

    def generator(self, start_char, total):
        '''
        get a sequence of integers from the model 
        '''
        h = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.charset_size, 1))
        x[self.char_to_idx[start_char]] = 1
        ixes = []
        for t in xrange(total):
            h = np.tanh(np.dot(self.w_i2h, x) + np.dot(self.w_h2h, h) + self.b_h)
            y = np.dot(self.w_h2o, h) + self.b_o
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.charset_size), p=p.ravel())
            x = np.zeros((self.charset_size, 1))
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(self.idx_to_char[ix] for ix in ixes)
        print 'generator output: \n', txt

    def save_model(self, model_name):
        with open(model_name, 'wb') as f:
            data = [self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o, 
                self.hidden_size, self.charset_size,
                self.char_to_idx, self.idx_to_char
                ]
            pickle.dump(data, f)

    def load_model(self, model_name):
        with open(model_name, 'rb') as f:
            data = pickle.load(f)
            self.w_i2h, self.w_h2h, self.w_h2o, self.b_h, self.b_o,\
            self.hidden_size, self.charset_size,\
            self.char_to_idx, self.idx_to_char = data
