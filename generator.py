from char_rnn import char_rnn

if __name__ == '__main__':
    rnn = char_rnn()
    rnn.load_model('test.model')
    rnn.generator('a', 1000)
