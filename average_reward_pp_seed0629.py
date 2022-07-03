import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
#seed_MAAC_PP5

step_size= -1
directory_seeds = "./results_PP/MDAC_PP7_seed/"

pd_reward_mean_seed2 = pd.read_csv(directory_seeds+'train_score_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed2 = pd.read_csv(directory_seeds+'train_scorestd_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed3 = pd.read_csv(directory_seeds+'train_score_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed3 = pd.read_csv(directory_seeds+'train_scorestd_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed4 = pd.read_csv(directory_seeds+'train_score_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed4 = pd.read_csv(directory_seeds+'train_scorestd_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed5 = pd.read_csv(directory_seeds+'train_score_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed5 = pd.read_csv(directory_seeds+'train_scorestd_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed6 = pd.read_csv(directory_seeds+'train_score_seed_6.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed6 = pd.read_csv(directory_seeds+'train_scorestd_seed_6.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed8 = pd.read_csv(directory_seeds+'train_score_seed_8.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed8 = pd.read_csv(directory_seeds+'train_scorestd_seed_8.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed10 = pd.read_csv(directory_seeds+'train_score_seed_10.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed10 = pd.read_csv(directory_seeds+'train_scorestd_seed_10.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed12 = pd.read_csv(directory_seeds+'train_score_seed_12.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed12 = pd.read_csv(directory_seeds+'train_scorestd_seed_12.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed15 = pd.read_csv(directory_seeds+'train_score_seed_15.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed15 = pd.read_csv(directory_seeds+'train_scorestd_seed_15.csv', delimiter=";", names=["Episode", "Score"])




x_seed2_pp5 = pd_reward_mean_seed2["Episode"].values[0:step_size]
mean_seed2_pp5 = pd_reward_mean_seed2["Score"].values[0:step_size]
std_seed2_pp5 = pd_reward_std_seed2["Score"].values[0:step_size]

x_seed3_pp5 = pd_reward_mean_seed3["Episode"].values[0:step_size]
mean_seed3_pp5 = pd_reward_mean_seed3["Score"].values[0:step_size]
std_seed3_pp5 = pd_reward_std_seed3["Score"].values[0:step_size]

x_seed4_pp5 = pd_reward_mean_seed4["Episode"].values[0:step_size]
mean_seed4_pp5 = pd_reward_mean_seed4["Score"].values[0:step_size]
std_seed4_pp5 = pd_reward_std_seed4["Score"].values[0:step_size]

x_seed5_pp5 = pd_reward_mean_seed5["Episode"].values[0:step_size]
mean_seed5_pp5 = pd_reward_mean_seed5["Score"].values[0:step_size]
std_seed5_pp5 = pd_reward_std_seed5["Score"].values[0:step_size]

x_seed6_pp5 = pd_reward_mean_seed6["Episode"].values[0:step_size]
mean_seed6_pp5 = pd_reward_mean_seed6["Score"].values[0:step_size]
std_seed6_pp5 = pd_reward_std_seed6["Score"].values[0:step_size]

x_seed8_pp5 = pd_reward_mean_seed8["Episode"].values[0:step_size]
mean_seed8_pp5 = pd_reward_mean_seed8["Score"].values[0:step_size]
std_seed8_pp5 = pd_reward_std_seed8["Score"].values[0:step_size]

x_seed10_pp5 = pd_reward_mean_seed10["Episode"].values[0:step_size]
mean_seed10_pp5 = pd_reward_mean_seed10["Score"].values[0:step_size]
std_seed10_pp5 = pd_reward_std_seed10["Score"].values[0:step_size]


x_seed12_pp5 = pd_reward_mean_seed12["Episode"].values[0:step_size]
mean_seed12_pp5 = pd_reward_mean_seed12["Score"].values[0:step_size]
std_seed12_pp5 = pd_reward_std_seed12["Score"].values[0:step_size]

x_seed15_pp5 = pd_reward_mean_seed15["Episode"].values[0:step_size]
mean_seed15_pp5 = pd_reward_mean_seed15["Score"].values[0:step_size]
std_seed15_pp5 = pd_reward_std_seed15["Score"].values[0:step_size]





#seed_test_DOMAC
directory_seeds2 = "./results_PP/MDOMAC_pp5_seed/"

pd_reward_mean_seed2a = pd.read_csv(directory_seeds2+'train_score_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed2a = pd.read_csv(directory_seeds2+'train_scorestd_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed3a = pd.read_csv(directory_seeds2+'train_score_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed3a = pd.read_csv(directory_seeds2+'train_scorestd_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed4a = pd.read_csv(directory_seeds2+'train_score_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed4a = pd.read_csv(directory_seeds2+'train_scorestd_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed5a = pd.read_csv(directory_seeds2+'train_score_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed5a = pd.read_csv(directory_seeds2+'train_scorestd_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed6a = pd.read_csv(directory_seeds2+'train_score_seed_6.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed6a = pd.read_csv(directory_seeds2+'train_scorestd_seed_6.csv', delimiter=";", names=["Episode", "Score"])

pd_reward_mean_seed8a = pd.read_csv(directory_seeds2+'train_score_seed_8.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed8a = pd.read_csv(directory_seeds2+'train_scorestd_seed_8.csv', delimiter=";", names=["Episode", "Score"])

pd_reward_mean_seed10a = pd.read_csv(directory_seeds2+'train_score_seed_10.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed10a = pd.read_csv(directory_seeds2+'train_scorestd_seed_10.csv', delimiter=";", names=["Episode", "Score"])

pd_reward_mean_seed12a = pd.read_csv(directory_seeds2+'train_score_seed_12.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed12a = pd.read_csv(directory_seeds2+'train_scorestd_seed_12.csv', delimiter=";", names=["Episode", "Score"])


pd_reward_mean_seed15a = pd.read_csv(directory_seeds2+'train_score_seed_15.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed15a = pd.read_csv(directory_seeds2+'train_scorestd_seed_15.csv', delimiter=";", names=["Episode", "Score"])




x_seed2a_pp5 = pd_reward_mean_seed2a["Episode"].values[0:step_size]
mean_seed2a_pp5 = pd_reward_mean_seed2a["Score"].values[0:step_size]
std_seed2a_pp5 = pd_reward_std_seed2a["Score"].values[0:step_size]

x_seed3a_pp5 = pd_reward_mean_seed3a["Episode"].values[0:step_size]
mean_seed3a_pp5 = pd_reward_mean_seed3a["Score"].values[0:step_size]
std_seed3a_pp5 = pd_reward_std_seed3a["Score"].values[0:step_size]

x_seed4a_pp5 = pd_reward_mean_seed4a["Episode"].values[0:step_size]
mean_seed4a_pp5 = pd_reward_mean_seed4a["Score"].values[0:step_size]
std_seed4a_pp5 = pd_reward_std_seed4a["Score"].values[0:step_size]

x_seed5a_pp5 = pd_reward_mean_seed5a["Episode"].values[0:step_size]
mean_seed5a_pp5 = pd_reward_mean_seed5a["Score"].values[0:step_size]
std_seed5a_pp5 = pd_reward_std_seed5a["Score"].values[0:step_size]

x_seed6a_pp5 = pd_reward_mean_seed6a["Episode"].values[0:step_size]
mean_seed6a_pp5 = pd_reward_mean_seed6a["Score"].values[0:step_size]
std_seed6a_pp5 = pd_reward_std_seed6a["Score"].values[0:step_size]

x_seed8a_pp5 = pd_reward_mean_seed8a["Episode"].values[0:step_size]
mean_seed8a_pp5 = pd_reward_mean_seed8a["Score"].values[0:step_size]
std_seed8a_pp5 = pd_reward_std_seed8a["Score"].values[0:step_size]


x_seed10a_pp5 = pd_reward_mean_seed10a["Episode"].values[0:step_size]
mean_seed10a_pp5 = pd_reward_mean_seed10a["Score"].values[0:step_size]
std_seed10a_pp5 = pd_reward_std_seed10a["Score"].values[0:step_size]

x_seed12a_pp5 = pd_reward_mean_seed12a["Episode"].values[0:step_size]
mean_seed12a_pp5 = pd_reward_mean_seed12a["Score"].values[0:step_size]
std_seed12a_pp5 = pd_reward_std_seed12a["Score"].values[0:step_size]

x_seed15a_pp5 = pd_reward_mean_seed15a["Episode"].values[0:step_size]
mean_seed15a_pp5 = pd_reward_mean_seed15a["Score"].values[0:step_size]
std_seed15a_pp5 = pd_reward_std_seed15a["Score"].values[0:step_size]



#testPP4v2_MAACseeds
directory_seeds_pp7 = "./results_PP/MAAC_pp7_seed/"

pd_reward_mean_seed2b = pd.read_csv(directory_seeds_pp7+'train_score_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed2b = pd.read_csv(directory_seeds_pp7+'train_scorestd_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed3b = pd.read_csv(directory_seeds_pp7+'train_score_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed3b = pd.read_csv(directory_seeds_pp7+'train_scorestd_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed4b = pd.read_csv(directory_seeds_pp7+'train_score_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed4b = pd.read_csv(directory_seeds_pp7+'train_scorestd_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed5b = pd.read_csv(directory_seeds_pp7+'train_score_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed5b = pd.read_csv(directory_seeds_pp7+'train_scorestd_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed6b = pd.read_csv(directory_seeds_pp7+'train_score_seed_6.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed6b = pd.read_csv(directory_seeds_pp7+'train_scorestd_seed_6.csv', delimiter=";", names=["Episode", "Score"])

x_seed2b_pp5 = pd_reward_mean_seed2b["Episode"].values[0:step_size]
mean_seed2b_pp5 = pd_reward_mean_seed2b["Score"].values[0:step_size]
std_seed2b_pp5 = pd_reward_std_seed2b["Score"].values[0:step_size]

x_seed3b_pp5 = pd_reward_mean_seed3b["Episode"].values[0:step_size]
mean_seed3b_pp5 = pd_reward_mean_seed3b["Score"].values[0:step_size]
std_seed3b_pp5 = pd_reward_std_seed3b["Score"].values[0:step_size]

x_seed4b_pp5 = pd_reward_mean_seed4b["Episode"].values[0:step_size]
mean_seed4b_pp5 = pd_reward_mean_seed4b["Score"].values[0:step_size]
std_seed4b_pp5 = pd_reward_std_seed4b["Score"].values[0:step_size]

x_seed5b_pp5 = pd_reward_mean_seed5b["Episode"].values[0:step_size]
mean_seed5b_pp5 = pd_reward_mean_seed5b["Score"].values[0:step_size]
std_seed5b_pp5 = pd_reward_std_seed5b["Score"].values[0:step_size]

x_seed6b_pp5 = pd_reward_mean_seed6b["Episode"].values[0:step_size]
mean_seed6b_pp5 = pd_reward_mean_seed6b["Score"].values[0:step_size]
std_seed6b_pp5 = pd_reward_std_seed6b["Score"].values[0:step_size]
#seed_test_DOMAC
directory_seeds_pp7a = "./results_PP/MDOMAC_pp7_seed/"

pd_reward_mean_seed2c = pd.read_csv(directory_seeds_pp7a+'train_score_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed2c = pd.read_csv(directory_seeds_pp7a+'train_scorestd_seed_2.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed3c = pd.read_csv(directory_seeds_pp7a+'train_score_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed3c = pd.read_csv(directory_seeds_pp7a+'train_scorestd_seed_3.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed4c = pd.read_csv(directory_seeds_pp7a+'train_score_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed4c = pd.read_csv(directory_seeds_pp7a+'train_scorestd_seed_4.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed5c = pd.read_csv(directory_seeds_pp7a+'train_score_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed5c = pd.read_csv(directory_seeds_pp7a+'train_scorestd_seed_5.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_mean_seed6c = pd.read_csv(directory_seeds_pp7a+'train_score_seed_6.csv', delimiter=";", names=["Episode", "Score"])
pd_reward_std_seed6c = pd.read_csv(directory_seeds_pp7a+'train_scorestd_seed_6.csv', delimiter=";", names=["Episode", "Score"])

x_seed2c_pp7 = pd_reward_mean_seed2c["Episode"].values[0:step_size]
mean_seed2c_pp7 = pd_reward_mean_seed2c["Score"].values[0:step_size]
std_seed2c_pp7 = pd_reward_std_seed2c["Score"].values[0:step_size]

x_seed3c_pp7 = pd_reward_mean_seed3c["Episode"].values[0:step_size]
mean_seed3c_pp7 = pd_reward_mean_seed3c["Score"].values[0:step_size]
std_seed3c_pp7 = pd_reward_std_seed3c["Score"].values[0:step_size]

x_seed4c_pp7 = pd_reward_mean_seed4c["Episode"].values[0:step_size]
mean_seed4c_pp7 = pd_reward_mean_seed4c["Score"].values[0:step_size]
std_seed4c_pp7 = pd_reward_std_seed4c["Score"].values[0:step_size]

x_seed5c_pp7 = pd_reward_mean_seed5c["Episode"].values[0:step_size]
mean_seed5c_pp7 = pd_reward_mean_seed5c["Score"].values[0:step_size]
std_seed5c_pp7 = pd_reward_std_seed5c["Score"].values[0:step_size]

x_seed6c_pp7 = pd_reward_mean_seed6c["Episode"].values[0:step_size]
mean_seed6c_pp7 = pd_reward_mean_seed6c["Score"].values[0:step_size]
std_seed6c_pp7 = pd_reward_std_seed6c["Score"].values[0:step_size]

import matplotlib.ticker as ticker


scale=4
#plot the figure
#fig, axs = plt.subplots(2,2,figsize=(scale*1.618,scale))
fig, axs = plt.subplots(figsize=(scale*1.618,scale))
#print('x_lable_pomm',x_lable_pomm)
axs.xaxis.set_major_formatter(ticker.EngFormatter())
base_line_seed2, = axs.plot(x_seed2_pp5, mean_seed2_pp5, 'gray', label= "seed2")
base_line_seed3, = axs.plot(x_seed3_pp5, mean_seed3_pp5, '#FC5A50', label= "seed3")
base_line_seed4, = axs.plot(x_seed4_pp5, mean_seed4_pp5, 'steelblue', label= "seed4")
base_line_seed5, = axs.plot(x_seed5_pp5, mean_seed5_pp5, 'seagreen', label= "seed5")
base_line_seed6, = axs.plot(x_seed6_pp5, mean_seed6_pp5, '#808000', label= "seed6")
base_line_seed8, = axs.plot(x_seed8_pp5, mean_seed8_pp5, 'rosybrown', label= "seed8")
base_line_seed10, = axs.plot(x_seed10_pp5, mean_seed10_pp5, '#A0522D', label= "seed10")
base_line_seed12, = axs.plot(x_seed12_pp5, mean_seed12_pp5, 'orange', label= "seed12")
base_line_seed15, = axs.plot(x_seed15_pp5, mean_seed15_pp5, 'mediumpurple', label= "seed15")


axs.fill_between(x_seed2_pp5, mean_seed2_pp5 - std_seed2_pp5, mean_seed2_pp5 + std_seed2_pp5, color='gray', alpha=0.1)
axs.fill_between(x_seed3_pp5, mean_seed3_pp5 - std_seed3_pp5, mean_seed3_pp5 + std_seed3_pp5, color='#FC5A50', alpha=0.1)
axs.fill_between(x_seed4_pp5, mean_seed4_pp5 - std_seed4_pp5, mean_seed4_pp5 + std_seed4_pp5, color='steelblue', alpha=0.1)
axs.fill_between(x_seed5_pp5, mean_seed5_pp5 - std_seed5_pp5, mean_seed5_pp5 + std_seed5_pp5, color='seagreen', alpha=0.1)
axs.fill_between(x_seed6_pp5, mean_seed6_pp5 - std_seed6_pp5, mean_seed6_pp5 + std_seed6_pp5, color='#808000', alpha=0.1)
axs.fill_between(x_seed8_pp5, mean_seed8_pp5 - std_seed8_pp5, mean_seed8_pp5 + std_seed8_pp5, color='rosybrown', alpha=0.1)
axs.fill_between(x_seed10_pp5, mean_seed10_pp5 - std_seed10_pp5, mean_seed10_pp5 + std_seed10_pp5, color='#A0522D', alpha=0.1)
axs.fill_between(x_seed12_pp5, mean_seed12_pp5 - std_seed12_pp5, mean_seed12_pp5 + std_seed12_pp5, color='orange', alpha=0.1)
axs.fill_between(x_seed15_pp5, mean_seed15_pp5 - std_seed15_pp5, mean_seed15_pp5 + std_seed15_pp5, color='mediumpurple', alpha=0.1)





axs.legend(fontsize=8, loc = 'lower right')
axs.xaxis.set_major_formatter(ticker.EngFormatter())
axs.set(ylabel='Average Return')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

axs.set(xlabel='Episode')
#axs.get_xaxis().set_visible(False)
axs.yaxis.label.set_size(15)
axs.grid(True)
#axs.set_xticklabels([])
axs.xaxis.label.set_size(15)
fig.tight_layout()
#plt.savefig('Averaged Reward-Pomm-comateam.pdf')
plt.savefig('Averaged Reward-PP4v2_DMAC_seed.png')
#plt.legend(loc='lower right')
plt.show()