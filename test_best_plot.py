
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv('results-winner-gains.csv')

sns.boxplot(data=df, x='enemy', y='gain', hue='EA').set_title('gain of best solutions')
sns.despine(offset=10, trim=True)

plt.xticks([0,1], ['[1,2,4,7]', '[3,6,7,8]'])
plt.xlabel('training set')
plt.ylabel('total gain')
plt.tight_layout()
plt.savefig('gains.png')
