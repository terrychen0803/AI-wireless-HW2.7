import numpy as np
import tensorflow.compat.v1 as tf
import scipy.io as sio

tf.disable_v2_behavior()
np.random.seed(1)
tf.set_random_seed(1)

# import our problems, networks and training modules
from tools import problems, networks, train, raputil

K = 64
mu = 2
SNR_train = [5, 10, 15, 20, 25, 30, 35, 40]
training_epochs = 200
batch_size = 50
ce_type = 'mmse'  # channel estimation: 'ls', 'mmse', 'dnn'
test_ce = True
CP_flag = False

BER = []
prob = []
x_hat_T = []
sess, input_holder, output = [], [], []
MSE_T, MSE_F = [], []

for i in range(0, 8):
    print("\nSNR=", SNR_train[i])

    if ce_type == 'dnn':
        sess, input_holder, output = networks.build_ce_dnn(
            K,
            SNR_train[i],
            training_epochs=training_epochs,
            batch_size=batch_size,
            savefile='dnn_ce/CE_DNN_' + ('CPFREE_' if CP_flag is False else '') +
                     str(2 ** mu) + 'QAM_SNR_' + str(SNR_train[i]) + 'dB.npz',
            test_flag=test_ce,
            cp_flag=CP_flag,
            nh1=500,
            nh2=250,
        )

    if test_ce:
        mse_t, mse_f = raputil.test_ce(
            sess,
            input_holder,
            output,
            SNR_train[i],
            est_type=ce_type,
            CP_flag=CP_flag,
        )
        MSE_T.append(mse_t)
        MSE_F.append(mse_f)

    tf.reset_default_graph()

print('BER', BER)
BER_matlab = np.array(BER)
print('MSE_T', MSE_T)
print('MSE_F', MSE_F)

savefile = 'MSE_' + ce_type + '_' + str(2 ** mu) + 'QAM' + ('_CP_FREE' if CP_flag is False else '')
if test_ce:
    sio.savemat(savefile + '.mat', {savefile: np.array(MSE_F)})
