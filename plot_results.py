import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv('neat-results.csv')
df['gen'] = df['gen'].apply(lambda x: x + 1)
df['EA'] = 'neat'

df2 = pd.read_csv('neuro-results.csv')
df2['gen'] = df2['gen'].apply(lambda x: x + 1)
df2['EA'] = 'neuro'

enemy0 = pd.concat([df.loc[df['enemy_group'] == 0], df2.loc[df2['enemy_group'] == 0]])
enemy1 = pd.concat([df.loc[df['enemy_group'] == 1], df2.loc[df2['enemy_group'] == 1]])

ax = sns.lineplot(data=enemy0, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemies [1,2,4,7]')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('results-enemy-group-0.png')

plt.cla()

ax = sns.lineplot(data=enemy1, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemies [3,6,7,8]')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('results-enemy-group-1.png')
