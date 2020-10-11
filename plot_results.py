import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)

df = pd.read_csv('neat-results.csv')
df['gen'] = df['gen'].apply(lambda x: x + 1)
df['EA'] = 'neat'

df2 = pd.read_csv('neuro-results.csv')
df2['gen'] = df2['gen'].apply(lambda x: x + 1)
df2['EA'] = 'neuro'

enemy0 = pd.concat([df.loc[df['enemy_group'] == 0], df2.loc[df2['enemy_group'] == 0]])
enemy1 = pd.concat([df.loc[df['enemy_group'] == 1], df2.loc[df2['enemy_group'] == 1]])

ax = sns.lineplot(data=enemy0, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 2')
plt.show()
ax = sns.lineplot(data=enemy1, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 7')
plt.show()
