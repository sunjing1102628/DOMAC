import numpy as np
import matplotlib.pyplot as plt
import os

gamma = 0.95
lr_a1=0.01
lr_maac_opp= 0.02
lr_madac= 0.03
lr_madac_opp= 0.04
#lr_critic=1e-3
# plot parameters
show = True
save = False
log_type = 'training'  # 'training', 'validation'
plot_step_size_training = 1
plot_step_size_validation = 1

file1 = r'./log/training_log_{}{}.npy'.format(gamma, lr_a1)
file2 = r'./log/training_log_{}{}.npy'.format(gamma, lr_maac_opp)
file3 = r'./log/training_log_{}{}.npy'.format(gamma, lr_madac)
file4 = r'./log/training_log_{}{}.npy'.format(gamma, lr_madac_opp)

log1 = np.load(file1,allow_pickle=True)
log2 = np.load(file2,allow_pickle=True)
log3 = np.load(file3,allow_pickle=True)
log4 = np.load(file4,allow_pickle=True)
# print('log is',len(log))
#
# print('log1 is',log)
# print('log1_len',len(log))
# print('1',log.shape[0])
# print('2',log[:log.shape[0] // plot_step_size_training * plot_step_size_training])
# print('3',len(log[:log.shape[0] // plot_step_size_training * plot_step_size_training]))
obj1 = log1[:log1.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log1.shape[0] // plot_step_size_training, -1).mean(axis=1)
# print('obj is',obj)
obj2 = log2[:log2.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log2.shape[0] // plot_step_size_training, -1).mean(axis=1)
# # print('obj is',obj)
obj3 = log3[:log3.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log3.shape[0] // plot_step_size_training, -1).mean(axis=1)
#print('obj3 is',obj3)
obj4 = log4[:log4.shape[0] // plot_step_size_training * plot_step_size_training].reshape(log4.shape[0] // plot_step_size_training, -1).mean(axis=1)
#print('obj4 is',obj4)
# print('4',range(obj.shape[0]))
# plot objective...

plt.xlabel('episode*100'.format(plot_step_size_training))
plt.ylabel('average reward')
plt.plot([_ for _ in range(obj1.shape[0])], obj1, color='tab:blue',label= 'maac')
plt.plot([_ for _ in range(obj2.shape[0])], obj2, color='tab:red',label= 'maac_opp')
plt.plot([_ for _ in range(obj3.shape[0])], obj3, color='tab:gray',label= 'maac_QR')
plt.plot([_ for _ in range(obj4.shape[0])], obj4, color='tab:purple',label= 'maac_QR_opp')
plt.grid()
plt.legend(loc = 'lower right')
plt.tight_layout()
if save:
    plt.savefig('./curves/{}_plt_'
                    .format(log_type)  # log type
                    + file1 + '.png')
if show:
    plt.show()
plt.close()

