import numpy as np
import matplotlib.pyplot as plot

y_data = np.loadtxt("symmetric_data.csv", delimiter=",") 
y_data_asymm = np.loadtxt("asymmetric_data.csv", delimiter=",") 
y_data_asymm2 = np.loadtxt("asymmetric_data_2.csv", delimiter=",") 

plot.figure()
plot.imshow(y_data)
plot.colorbar()
plot.savefig('plots/symmetric_data.png')

plot.figure()
plot.imshow(y_data_asymm)
plot.colorbar()
plot.savefig('plots/asymmetric_data.png')

plot.figure()
plot.imshow(y_data_asymm2)
plot.colorbar()
plot.savefig('plots/asymmetric_data_2.png')
