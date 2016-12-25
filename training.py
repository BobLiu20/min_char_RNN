import signal
from char_rnn import char_rnn

def sigint_handler(signum, frame):
    global rnn
    if rnn:
        print '\nsaving test model to test.model...'
        rnn.save_model('test.model')
    exit()

rnn = None

if __name__ == '__main__':
    # save model when ctrl+C or kill me.
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    rnn = char_rnn('test.txt')
    rnn.training()
