import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag

######Parameters###################
init_eta = 0.005
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 100000
nruns = 200
num_inputs = 4
num_outputs = 7
num_hidden = 4
termination_thresh = 0.0001 # stop at this loss
###################################



x_data = np.eye(4)
y_data = np.array(
    [[1, 1, 0, 0.5, 0, 0.5, 0],
     [1, 1, 0, 0, 0.5, 0, 0.5],
     [1, 0, 1, 0.5, 0, 0.5, 0],
     [1, 0, 1, 0, 0.5, 0, 0.5]])
U, S, V, = np.linalg.svd(y_data)
#y_data_asymm = np.zeros_like(y_data)
#y_data_asymm[range(len(S)), range(len(S))] = S
y_data_asymm = np.loadtxt("asymmetric_data.csv", delimiter=",")
U, S, V, = np.linalg.svd(y_data_asymm)

for rseed in xrange(0, nruns):
    for nonlinear in [True, False]:
        nonlinearity_function = tf.nn.leaky_relu
        for nlayer in [4, 3, 2]:
            for symmetric in [False, True]:
                num_hidden = num_hidden
                print "nlayer %i nonlinear %i symmetric %i run %i" % (nlayer, nonlinear, symmetric, rseed)
                filename_prefix = "results/nlayer_%i_nonlinear_%i_symmetric_%i_rseed_%i_" %(nlayer,nonlinear,symmetric,rseed)

                np.random.seed(rseed)
                tf.set_random_seed(rseed)
                this_x_data = x_data
                this_y_data = y_data if symmetric else y_data_asymm

                input_ph = tf.placeholder(tf.float32, shape=[None, num_inputs])
                target_ph = tf.placeholder(tf.float32, shape=[None, num_outputs])
                if nonlinear:
                    if nlayer == 2:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = nonlinearity_function(tf.matmul(W1,tf.transpose(input_ph)))
                        pre_output = tf.matmul(W2,internal_rep)
                    elif nlayer == 3:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = nonlinearity_function(tf.matmul(W1,tf.transpose(input_ph)))
                            
                        pre_output = tf.matmul(W3,nonlinearity_function(tf.matmul(W2,internal_rep)))
                    elif nlayer == 4:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W4 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = nonlinearity_function(tf.matmul(W1,tf.transpose(input_ph)))
                            
                        pre_output = tf.matmul(W4,nonlinearity_function(tf.matmul(W3,nonlinearity_function(tf.matmul(W2,internal_rep)))))
                    else:
                        print "Error, invalid number of layers given"
                        exit(1)
                else:
                    if nlayer == 2:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = (tf.matmul(W1,tf.transpose(input_ph)))
                        pre_output = tf.matmul(W2,internal_rep)
                    elif nlayer == 3:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = (tf.matmul(W1,tf.transpose(input_ph)))
                            
                        pre_output = tf.matmul(W3,(tf.matmul(W2,internal_rep)))
                    elif nlayer == 4:
                        W1 = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,1./(num_hidden+num_inputs)))
                        W2 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W3 = tf.Variable(tf.random_uniform([num_hidden,num_hidden],0.,1./num_hidden))
                        W4 = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,1./(num_hidden+num_outputs)))
                        internal_rep = (tf.matmul(W1,tf.transpose(input_ph)))
                            
                        pre_output = tf.matmul(W4,(tf.matmul(W3,(tf.matmul(W2,internal_rep)))))
                    else:
                        print "Error, invalid number of layers given"
                        exit(1)

                output = nonlinearity_function(pre_output)

                loss = tf.reduce_sum(tf.square(output - tf.transpose(target_ph)))# +0.05*(tf.nn.l2_loss(internal_rep))
                output_grad = tf.gradients(loss,[output])[0]
                eta_ph = tf.placeholder(tf.float32)
                optimizer = tf.train.GradientDescentOptimizer(eta_ph)
                train = optimizer.minimize(loss)

                init = tf.global_variables_initializer()

                sess = tf.Session()
                sess.run(init)

                def test_accuracy():
                    MSE = sess.run(loss,
                                   feed_dict={input_ph: this_x_data,target_ph: this_y_data})
                    MSE /= num_inputs 
                    return MSE

                def print_outputs():
                    print sess.run(output,feed_dict={input_ph: this_x_data})


                def print_preoutputs():
                    print sess.run(pre_output,feed_dict={input_ph: this_x_data})


                def save_activations(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        res = sess.run(tf_object, feed_dict={input_ph: this_x_data})
                        np.savetxt(fout, res, delimiter=',')


                def save_weights(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        np.savetxt(fout,sess.run(tf_object),delimiter=',')


                def run_train_epoch():
                    sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: this_x_data,target_ph: this_y_data})

                print "Initial MSE: %f" %(test_accuracy())

                #loaded_pre_outputs = np.loadtxt(pre_output_filename_to_load,delimiter=',')

                curr_eta = init_eta
                rep_track = []
                loss_filename = filename_prefix + "loss_track.csv"
                with open(loss_filename, 'w') as fout:
                    fout.write("epoch, MSE\n")
                    curr_mse = test_accuracy()
                    fout.write("%i, %f\n" %(0, curr_mse))
                    for epoch in xrange(nepochs):
                        run_train_epoch()
                        if epoch % 5 == 0:
                            curr_mse = test_accuracy()
                            print "epoch: %i, MSE: %f" %(epoch, curr_mse)	
                            fout.write("%i, %f\n" %(epoch, curr_mse))
                            if curr_mse < termination_thresh:
                                print("Early stop!")
                                break
#                            if epoch % 100 == 0:
#                                save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
#                                save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
                        
                        if epoch % eta_decay_epoch == 0:
                            curr_eta *= eta_decay
                    
                print "Final MSE: %f" %(test_accuracy())
