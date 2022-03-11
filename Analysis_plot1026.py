import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

directory1 = "./results/final_results72new_2.5e45e4_0.95acc2/"
#directory1 = "./results/final_results51new1/"
directory2 = "./results/final_results74new_2.5e45e4_0.95acc2/"
#directory2 = "./results/final_results54new5e51e4/"
directory3 = "./results/final_results73new_2.5e45e4_0.95ids/"
directory4 = "./results/final_results72new_2.5e45e4_0.95/"
# directory4 = "./results/final_results54new_3e4_0.95_clip10/"
directory5 = "./results/final_results71new_2.5e45e4_0.95ids2/"
directory6 = "./results/final_results74new_2.5e45e4_0.95ids/"

train_files = glob.glob(directory1 + 'train*.csv')
test_files = glob.glob(directory2 + 'train*.csv')
madac_files = glob.glob(directory3 + 'train*.csv')
madacopp_files = glob.glob(directory4 + 'train*.csv')
madacopp1_files = glob.glob(directory5 + 'train*.csv')
madacopp2_files = glob.glob(directory6 + 'train*.csv')


#file_results03
pd_madac = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for madac_file in madac_files:
    pd_madac_file = pd.read_csv(madac_file, delimiter=";", names=["Episode", "Score"])
    pd_madac_file["Seed"] = madac_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_madac = pd.concat([pd_madac, pd_madac_file])
    pd_madac.reset_index()

pd_madac = pd_madac.reset_index(drop=True)

pd_madac_gb = pd_madac.groupby("Episode")

pd_madac_mean = pd_madac_gb.mean()
pd_madac_mean = pd_madac_mean.reset_index()
pd_madac_mean = pd_madac_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_madac_var = pd_madac_gb.var()
pd_madac_var = pd_madac_var.reset_index()
pd_madac_var = pd_madac_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_madac = pd_madac_mean["Episode"].values
mean_madac = pd_madac_mean["MeanScore"].values
std_madac = np.sqrt(pd_madac_var["VarScore"].values)



#file_results04
pd_madacopp = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for madacopp_file in madacopp_files:
    pd_madacopp_file = pd.read_csv(madacopp_file, delimiter=";", names=["Episode", "Score"])
    pd_madacopp_file["Seed"] = madacopp_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_madacopp = pd.concat([pd_madacopp, pd_madacopp_file])
    pd_madacopp.reset_index()

pd_madacopp = pd_madacopp.reset_index(drop=True)

pd_madacopp_gb = pd_madacopp.groupby("Episode")

pd_madacopp_mean = pd_madacopp_gb.mean()
pd_madacopp_mean = pd_madacopp_mean.reset_index()
pd_madacopp_mean = pd_madacopp_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_madacopp_var = pd_madacopp_gb.var()
pd_madacopp_var = pd_madacopp_var.reset_index()
pd_madacopp_var = pd_madacopp_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_madacopp = pd_madacopp_mean["Episode"].values
mean_madacopp = pd_madacopp_mean["MeanScore"].values
std_madacopp = np.sqrt(pd_madacopp_var["VarScore"].values)

#file_results01
pd_train = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for train_file in train_files:
    pd_train_file = pd.read_csv(train_file, delimiter=";", names=["Episode", "Score"])
    pd_train_file["Seed"] = train_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_train = pd.concat([pd_train, pd_train_file])
    pd_train.reset_index()

pd_train = pd_train.reset_index(drop=True)

pd_train_gb = pd_train.groupby("Episode")
#print('pd_train_gb',pd_train_gb)
pd_train_mean = pd_train_gb.mean()
#print('pd_train_mean',pd_train_mean)
pd_train_mean = pd_train_mean.reset_index()
pd_train_mean = pd_train_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_train_var = pd_train_gb.var()
pd_train_var = pd_train_var.reset_index()
pd_train_var = pd_train_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_train = pd_train_mean["Episode"].values
mean_train = pd_train_mean["MeanScore"].values
std_train = np.sqrt(pd_train_var["VarScore"].values)

#file_results02
pd_test = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for test_file in test_files:
    pd_test_file = pd.read_csv(test_file, delimiter=";", names=["Episode", "Score"])
    pd_test_file["Seed"] = test_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_test = pd.concat([pd_test, pd_test_file])
    pd_test.reset_index()

pd_test = pd_test.reset_index(drop=True)
pd_test_gb = pd_test.groupby("Episode")
#print('pd_test_gb',pd_test_gb)
pd_test_mean = pd_test_gb.mean()
pd_test_mean = pd_test_mean.reset_index()
pd_test_mean = pd_test_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_test_var = pd_test_gb.var()
pd_test_var = pd_test_var.reset_index()
pd_test_var = pd_test_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_test = pd_test_mean["Episode"].values
mean_test = pd_test_mean["MeanScore"].values
std_test = np.sqrt(pd_test_var["VarScore"].values)
#file_results06
pd_madacopp2 = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for madacopp2_file in madacopp2_files:
    pd_madacopp2_file = pd.read_csv(madacopp2_file, delimiter=";", names=["Episode", "Score"])
    pd_madacopp2_file["Seed"] = madacopp2_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_madacopp2 = pd.concat([pd_madacopp2, pd_madacopp2_file])
    pd_madacopp2.reset_index()

pd_madacopp2 = pd_madacopp2.reset_index(drop=True)

pd_madacopp2_gb = pd_madacopp2.groupby("Episode")
#print('pd_madacopp_gb',pd_madacopp_gb)
pd_madacopp2_mean = pd_madacopp_gb.mean()
pd_madacopp2_mean = pd_madacopp2_mean.reset_index()
pd_madacopp2_mean = pd_madacopp2_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_madacopp2_var = pd_madacopp2_gb.var()
pd_madacopp2_var = pd_madacopp2_var.reset_index()
pd_madacopp2_var = pd_madacopp2_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_madacopp2 = pd_madacopp2_mean["Episode"].values
mean_madacopp2 = pd_madacopp2_mean["MeanScore"].values
std_madacopp2 = np.sqrt(pd_madacopp2_var["VarScore"].values)
#file_results05
pd_madacopp1 = pd.DataFrame(columns = ["Episode", "Score", "Seed"])
for madacopp1_file in madacopp1_files:
    pd_madacopp1_file = pd.read_csv(madacopp1_file, delimiter=";", names=["Episode", "Score"])
    pd_madacopp1_file["Seed"] = madacopp1_file.split("/")[-1].split("_")[-1].split(".")[0]
    pd_madacopp1 = pd.concat([pd_madacopp1, pd_madacopp1_file])
    pd_madacopp1.reset_index()

pd_madacopp1 = pd_madacopp1.reset_index(drop=True)

pd_madacopp1_gb = pd_madacopp1.groupby("Episode")

pd_madacopp1_mean = pd_madacopp_gb.mean()
pd_madacopp1_mean = pd_madacopp1_mean.reset_index()
pd_madacopp1_mean = pd_madacopp1_mean.rename(columns={'Episode': 'Episode', 'Score': 'MeanScore'})
pd_madacopp1_var = pd_madacopp1_gb.var()
pd_madacopp1_var = pd_madacopp1_var.reset_index()
pd_madacopp1_var = pd_madacopp1_var.rename(columns={'Episode': 'Episode', 'Score': 'VarScore'})

x_madacopp1 = pd_madacopp1_mean["Episode"].values
mean_madacopp1 = pd_madacopp1_mean["MeanScore"].values
std_madacopp1 = np.sqrt(pd_madacopp1_var["VarScore"].values)

'''def custom_plot(x, y, z, xlabel, ylabel,title,color, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    #ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y, color)
    ax.fill_between(x, y - z/2, y + z/2, facecolor=base_line.get_color(), alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

custom_plot(x_test, mean_test, std_test, "# Episode", "Score", "Test score averaged over seeds".format(len(test_files)),"b",(10,5))
custom_plot(x_train, mean_train, std_train, "# Episode", "Score", "Train score averaged over seeds".format(len(train_files)), "r", (10,5))
'''
fig, ax = plt.subplots(figsize=(10,5))
print('x_madac',x_madac)
#base_line1, = ax.plot(x_train, mean_train, 'tab:blue', label= "maacopp_acc")
#base_line2, = ax.plot(x_test, mean_test, 'tab:red',label= 'maacoppQR_acc')
base_line3, = ax.plot(x_madac, mean_madac, "olive",label= "maac")
base_line4, = ax.plot(x_madacopp, mean_madacopp, "purple",label= "maac_opp")
#base_line5, = ax.plot(x_madacopp1, mean_madacopp1, "green",label= "maac_QR")
base_line6, = ax.plot(x_madacopp2, mean_madacopp2, "cyan",label= "maac_QRopp")
#ax.fill_between(x_train, mean_train - std_train/2, mean_train + std_train/2, facecolor=base_line1.get_color(), alpha=0.5)
#ax.fill_between(x_test, mean_test - std_test/2, mean_test + std_test/2, facecolor=base_line2.get_color(), alpha=0.5)
ax.fill_between(x_madac, mean_madac- std_madac/2, mean_madac + std_madac/2, facecolor=base_line3.get_color(), alpha=0.5)
ax.fill_between(x_madacopp, mean_madacopp- std_madacopp/2, mean_madacopp + std_madacopp/2, facecolor=base_line4.get_color(), alpha=0.5)
#ax.fill_between(x_madacopp1, mean_madacopp1- std_madacopp1/2, mean_madacopp1 + std_madacopp1/2, facecolor=base_line5.get_color(), alpha=0.5)
ax.fill_between(x_madacopp2, mean_madacopp2- std_madacopp2/2, mean_madacopp2 + std_madacopp2/2, facecolor=base_line6.get_color(), alpha=0.5)
plt.title("Train Reward averaged over seeds 7*7".format(len(train_files)))
#plt.title("Accuarcy of estimated opponent model over 7*7".format(len(train_files)))
plt.xlabel("# Episode")
plt.ylabel("Average reward")
plt.grid()
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.show()