import numpy as np
import tensorflow as tf

y_data = np.array(
    [[1, 1, 0, 0.5, 0, 0.5, 0],
     [1, 1, 0, 0, 0.5, 0, 0.5],
     [1, 0, 1, 0.5, 0, 0.5, 0],
     [1, 0, 1, 0, 0.5, 0, 0.5]])
U, S, V, = np.linalg.svd(y_data, full_matrices=False)

var_by_unit = np.sum(np.square(y_data), axis=0)
print(var_by_unit)

new_y = tf.get_variable("new_y", shape=y_data.shape,
#                        initializer=tf.zeros_initializer())
                        initializer=tf.constant_initializer(np.multiply(V, np.expand_dims(S, 1))))

org_SVD_mat = np.diag(S**2)

new_y_var_by_unit = tf.reduce_sum(tf.square(new_y), axis=0)
SVD_loss = tf.nn.l2_loss(tf.matmul(new_y, tf.transpose(new_y)) - org_SVD_mat) 
var_by_unit_loss = tf.nn.l2_loss(new_y_var_by_unit - var_by_unit)

def rat_loss(el1, el2, desired_rat):
    """Tries to force parameters to have a desired ratio"""
    return tf.square(tf.square(tf.log(el1/el2)) - np.log(desired_rat)**2) 


mode_1_losses = rat_loss(new_y[0, 1], new_y[0,2], 2) + rat_loss(new_y[0, 2], new_y[0,3], 2) + rat_loss(new_y[0, 3], new_y[0,4], 2) + rat_loss(new_y[0, 4], new_y[0,5], 2) + rat_loss(new_y[0, 5], new_y[0, 6], 2)
mode_2_losses = rat_loss(new_y[0, 3], new_y[0,4], 3) + rat_loss(new_y[0, 4], new_y[0,5], 3) + rat_loss(new_y[0, 5], new_y[0, 6], 3)
mode_3_losses = rat_loss(new_y[0, 3], new_y[0,4], 5) + rat_loss(new_y[0, 4], new_y[0,5], 5) + rat_loss(new_y[0, 5], new_y[0, 6], 5)

total_loss = var_by_unit_loss + SVD_loss + mode_1_losses + mode_2_losses + mode_3_losses

optimizer = tf.train.AdamOptimizer(0.01)
optimize = optimizer.minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("initial")
    print(sess.run(new_y))
    print(sess.run(total_loss))
    for i in range(1000):
        sess.run(optimize)
    print("final")
    optimized_y = sess.run(new_y)
    print(optimized_y)
    print(sess.run([total_loss, var_by_unit_loss, SVD_loss]))
    row_norms = np.sqrt(np.sum(np.square(optimized_y), axis=1))
    optimized_y /= np.expand_dims(row_norms, 1)
    optimized_y *= np.expand_dims(S, 1)
    print(optimized_y)


    np.savetxt("asymmetric_data.csv", optimized_y, delimiter=',')
