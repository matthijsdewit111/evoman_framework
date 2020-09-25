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

enemy2 = pd.concat([df.loc[df['enemy'] == 2], df2.loc[df2['enemy'] == 2]])
enemy7 = pd.concat([df.loc[df['enemy'] == 7], df2.loc[df2['enemy'] == 7]])
enemy8 = pd.concat([df.loc[df['enemy'] == 8], df2.loc[df2['enemy'] == 8]])

ax = sns.lineplot(data=enemy2, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 2')
plt.show()
sns.lineplot(data=enemy7, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 7')
plt.show()
sns.lineplot(data=enemy8, x='gen', y='value', hue='EA', style='metric')
ax.set(xlabel='generation', ylabel='fitness', title='enemy 8')
plt.show()
