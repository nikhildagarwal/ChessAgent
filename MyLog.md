# March 13, 2025
* Conducted experiments on 2400 games of Kasparov on a Simple Feed Forward NN, Feed Forward NN with Attention and an RNN
* Feed Forward with attention performed the best
* Will be collecting more data next on Magnus Carlsen and Bobby Fischer

# March 15, 2025
* Finished Training my first model on 1300 epochs on 860,000 samples.
* The MSE flattened at 0.0053
* Final model is a Feed Forward NN with an attention layer (64 heads)
* Training for this model took around 21 hours on my CPU

# March 17, 2025
* Had my first test with the chess agent playing itself. The game started out well with the agent opening professionally. However, as the game developed the moves started to make less and less sense
* Started another training loop with a slightly modified dataset that repeats some states more than others in an attempt to fill the gaps in scarcity