import numpy as np
import matplotlib.pyplot as plot

x_data = np.eye(4)
y_data = np.array(
    [[1, 1, 0, 0.5, 0, 0.5, 0],
     [1, 1, 0, 0, 0.5, 0, 0.5],
     [1, 0, 1, 0.5, 0, 0.5, 0],
     [1, 0, 1, 0, 0.5, 0, 0.5]])
U, S, V, = np.linalg.svd(y_data)
y_data_asymm = np.zeros_like(y_data)
y_data_asymm[range(len(S)), range(len(S))] = S

plot.figure()
plot.imshow(y_data)
plot.colorbar()
plot.savefig('plots/symmetric_data.png')

plot.figure()
plot.imshow(y_data_asymm)
plot.colorbar()
plot.savefig('plots/asymmetric_data.png')
