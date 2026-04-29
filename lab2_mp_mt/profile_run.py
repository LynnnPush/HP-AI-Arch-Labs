from lib import data
from tvb_par import simulate

W, D = data.tvb76_weights_lengths()
W = W.tolist()
D = D.tolist()
N = len(W)

simulate(W, D, N, 2, 0.05, 15.0, 4.0, chunksize=57)
