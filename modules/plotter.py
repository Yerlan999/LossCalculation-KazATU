import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

prisoed_name = "ПС Промзона - ПС Погорелово (Ц.1)"

os.chdir(prisoed_name)

current_df = pd.read_csv('Эпюра токов.csv').dropna(axis=1, how='all')
fig, axs = plt.subplots(len(current_df.columns), sharex=True, figsize=(20,20))
fig.suptitle('Эпюра токов')

for i, column_name in enumerate(current_df.columns):
    print(column_name)
    axs[i].plot(current_df.loc[:,column_name])
    axs[i].set_title(column_name)

plt.savefig('Эпюра токов.png')

os.chdir("..")
