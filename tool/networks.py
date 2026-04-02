"""
Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation

This file follows the original repo structure and only fills the missing TODO blocks.
Added comments marked with [ADDED] explain the parts completed here.
"""
import numpy as np
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tools.shrinkage as shrinkage
from .train import load_trainable_vars, save_trainable_vars
from .raputil import sample_gen
from tensorflow.keras.layers import Dense


def build_ce_dnn(K, SNR, savefile, learning_rate=1e-3, training_epochs=2000,
                 batch_size=50, nh1=500, nh2=250, test_flag=False, cp_flag=True):
    n_input = 2 * K + 2 * K  # yp and xp as input
    n_output = 2 * K

    # [ADDED] Completed TODO block in the original repo.
    nn_input = tf.placeholder(tf.float32, [None, n_input], name='nn_input')
    H_true = tf.placeholder(tf.float32, [None, n_output], name='H_true')  # label

    dense1 = Dense(nh1, activation='relu', name='dense1')(nn_input)
    dense2 = Dense(nh2, activation='relu', name='dense2')(dense1)
    output_layer = Dense(n_output, activation=None, name='dense_out')(dense2)

    tmp = output_layer - H_true
    tmp = tf.square(tmp)
    H_out = output_layer

    # Define loss and optimizer, minimize the l2 loss
    loss_ = tf.reduce_mean(tf.reduce_sum(tmp, axis=1))

    global_step = tf.Variable(0, trainable=False)
    decay_steps, lr_decay = 20000, 0.1
    lr_ = tf.train.exponential_decay(learning_rate, global_step, decay_steps, lr_decay, name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_).minimize(
        loss_, global_step, var_list=tf.trainable_variables()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    log = str(state.get('log', ''))
    print(log)

    if test_flag:
        return sess, nn_input, H_out

    test_step = 5
    loss_history = []
    save = {}  # for the best model

    val_ls, val_labels, val_Yp, val_Xp = sample_gen(batch_size * 100, SNR, training_flag=False, CP_flag=cp_flag)

    for epoch in range(training_epochs + 1):
        train_loss = 0.
        for _ in range(20):
            batch_ls, batch_labels, Yp, Xp = sample_gen(batch_size, SNR, training_flag=True, CP_flag=cp_flag)
            sample = np.concatenate((Yp, Xp), axis=1)  # (bs, 4K)
            _, loss = sess.run([optimizer, loss_], feed_dict={nn_input: sample, H_true: batch_labels})
            train_loss += loss

        sys.stdout.write('\repoch={epoch:<6d} loss={loss:.9f} on train set'.format(epoch=epoch, loss=train_loss))
        sys.stdout.flush()

        # validation
        if epoch % test_step == 0:
            sample = np.concatenate((val_Yp, val_Xp), axis=1)  # (bs, 4K)
            loss = sess.run(loss_, feed_dict={nn_input: sample, H_true: val_labels})
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')

            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()

            # for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)

            print("\nepoch={epoch:<6d} loss={loss:.9f} (best={best:.9f}) on test set".format(
                epoch=epoch, loss=loss, best=loss_best))

    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)

    log = log + '\nloss={loss:.9f} in {i} iterations best={best:.9f} in {j} iterations'.format(
        loss=loss, i=epoch, best=loss_best, j=loss_history.argmin() * test_step)
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    print('optimization finished')
    return sess, nn_input, H_out
