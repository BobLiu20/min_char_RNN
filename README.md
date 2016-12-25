# min_char_RNN
a simple character RNN implement.

### statement
* The lost of source code was wrote according to [karpathy's gist](https://gist.github.com/karpathy/d4dee566867f8291f086)
* You can get background knowledge from [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### what's this
1. These code will training a rnn mode to generate article.
2. A mode file will be created when you kill the process(eg: Ctrl+C).
3. You can load pre-training mode to do a generate test.

### how to use
* run ```python training.py``` to do a training. An example output is like below:
```
iter 2000, loss: 70.034620
iter 2100, loss: 69.157650
iter 2200, loss: 68.066171
iter 2300, loss: 67.368618
iter 2400, loss: 66.422001
iter 2500, loss: 65.549434
iter 2600, loss: 64.841401
iter 2700, loss: 64.127791
iter 2800, loss: 63.122929
iter 2900, loss: 62.588574
generator output: 
re pnome.
-all whelld beage pit theel pepore che pnebeit be the the nlod woror mat ll latkers fotiy soet
'lo bs
Touper theounn chas hef bure
I but of and teein tuea.:
Whaild faglait bard masr aseind C
iter 3000, loss: 62.092721
```
* run ```python generator.py``` will load model file to generate string.
