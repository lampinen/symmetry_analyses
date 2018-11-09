import numpy as np
import tensorflow as tf

y_data = np.array(
    [[1, 1, 0, 0.5, 0, 0.5, 0],
     [1, 1, 0, 0, 0.5, 0, 0.5],
     [1, 0, 1, 0.5, 0, 0.5, 0],
     [1, 0, 1, 0, 0.5, 0, 0.5]])
U, S, V, = np.linalg.svd(y_data, full_matrices=False)
S_mat = tf.constant(np.diag(S), dtype=tf.float32)

var_by_unit = np.sum(np.square(y_data), axis=0)
print(var_by_unit)


new_U = tf.get_variable("new_U", shape=U.shape,
                        initializer=tf.constant_initializer(U))

new_V = tf.get_variable("new_V", shape=V.shape,
                        initializer=tf.constant_initializer(V))

new_y = tf.matmul(new_U, tf.matmul(S_mat, new_V)) 

new_y_var_by_unit = tf.reduce_sum(tf.square(new_y), axis=0)
#SVD_loss = tf.nn.l2_loss(tf.matmul(new_y, tf.transpose(new_y)) - org_SVD_mat) 
var_by_unit_loss = tf.nn.l2_loss(new_y_var_by_unit - var_by_unit)

orthogonal_loss = tf.nn.l2_loss(tf.matmul(tf.transpose(new_U), new_U) - tf.eye(4)) + tf.nn.l2_loss(tf.matmul(new_V, tf.transpose(new_V)) -tf.eye(4))

def rat_loss(el1, el2, desired_rat):
    """Tries to force parameters to have a desired ratio"""
    return tf.square(tf.square(tf.log(el1/el2)) - np.log(desired_rat)**2) 

def diff_loss(el1, el2):
    """Tries to force elements to be different"""
    return 1-tf.clip_by_value(tf.reduce_sum(((el1 - el2))**2), 0, 1)

positivity_loss = tf.nn.l2_loss(new_y - tf.abs(new_y))
asymmetry_loss = 0.

for mode in range(3):
    for i in range(7):
        for j in range(i, 7):
            asymmetry_loss += diff_loss(new_V[mode, i], new_V[mode, j])
    for i in range(4):
        for j in range(i, 4):
            asymmetry_loss += diff_loss(new_U[mode, i], new_U[mode, j])


#mode_1_losses = rat_loss(new_y[0, 1], new_y[0,2], 2) + rat_loss(new_y[0, 2], new_y[0,3], 2) + rat_loss(new_y[0, 3], new_y[0,4], 2) + rat_loss(new_y[0, 4], new_y[0,5], 2) + rat_loss(new_y[0, 5], new_y[0, 6], 2)
#mode_2_losses = rat_loss(new_y[0, 3], new_y[0,4], 3) + rat_loss(new_y[0, 4], new_y[0,5], 3) + rat_loss(new_y[0, 5], new_y[0, 6], 3)
#mode_3_losses = rat_loss(new_y[0, 3], new_y[0,4], 5) + rat_loss(new_y[0, 4], new_y[0,5], 5) + rat_loss(new_y[0, 5], new_y[0, 6], 5)
#
#column_relationship_loss = 0.
#for i in range(7):
#    for j in range(i + 1, 7):
#        column_relationship_loss += 1-tf.clip_by_value(tf.reduce_sum(((new_y[:, i] - new_y[:, j]))**2), 0, 1)
#

#total_loss = var_by_unit_loss + 10 * SVD_loss + 0.2 * (mode_1_losses + mode_2_losses + mode_3_losses)
#total_loss = var_by_unit_loss + 5*SVD_loss + column_relationship_loss

total_loss = var_by_unit_loss + 5*orthogonal_loss + 5*positivity_loss + asymmetry_loss 

optimizer = tf.train.AdamOptimizer(0.0001)
optimize = optimizer.minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("initial")
    print(sess.run(new_y))
    print(sess.run(total_loss))
    for i in range(10000):
        sess.run(optimize)
        if i % 100 == 0:
            print(sess.run([total_loss, var_by_unit_loss, orthogonal_loss, positivity_loss, asymmetry_loss]))
    print("final")
    optimized_y = sess.run(new_y)
    print(optimized_y)
    print(sess.run([total_loss, var_by_unit_loss, orthogonal_loss, positivity_loss, asymmetry_loss]))
    new_U, _, new_V = np.linalg.svd(optimized_y, full_matrices=False)
    optimized_y = np.matmul(new_U, np.matmul(np.diag(S), new_V))
    print(optimized_y)

# whatever
#
#new_V = np.array(
#    [[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
#     [1, 1, 0, 0, -1.5, -2, 0],
#     [0, 0, -1, -1, 0, 0, 15./8]:
#     [1, 0, 0, 0, 0, 0, 0]])
#
#new_U = np.array(
#    [[4, 3, 2, 1],
#     [1, 0, 0, -4],
#     [0, 2, -3, 0],
#     [1, 0, 0, 0]]).transpose()
#
#new_V /= np.expand_dims(np.sqrt(np.sum(np.square(new_V), axis=1)), 1)
#new_U = new_U/np.expand_dims(np.sqrt(np.sum(np.square(new_U), axis=0)), 0)
#print(new_V)
#print(new_U)
#
#y_data_asym = np.matmul(np.multiply(new_U, np.expand_dims(S, 1)), new_V)
#
#print(y_data_asym)
#print()
#
#
#
#U, S, V, = np.linalg.svd(y_data_asym, full_matrices=False)
#print(U)
#print(S)
#print(V)

np.savetxt("asymmetric_data.csv", y_data_asym, delimiter=',')
