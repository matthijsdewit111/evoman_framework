
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

df = pd.read_csv('results-winner-gains.csv')

df_NEAT = df[df.EA == "neat"]
df_Neuro = df[df.EA == "neuro"]
df1_NEAT = df[(df.enemy==0) & (df.EA == "neat")]
df1_Neuro = df[(df.enemy==0) & (df.EA == "neuro")]
df2_NEAT = df[(df.enemy==1) & (df.EA == "neat")]
df2_Neuro = df[(df.enemy==1) & (df.EA == "neuro")]

import scipy
#Comparing results of both algorithms for both groups of enemies
value, pvalue = scipy.stats.ks_2samp(df_NEAT['gain'].values, df_Neuro['gain'].values)
print(value, pvalue)
#Comparing results of both algorithms for first group of enemies
value1, pvalue1 = scipy.stats.ks_2samp(df1_NEAT['gain'].values, df1_Neuro['gain'].values)
print(value1, pvalue1)
#Comparing results of both algorithms for second group of enemies
value2, pvalue2 = scipy.stats.ks_2samp(df2_NEAT['gain'].values, df2_Neuro['gain'].values)
print(value2, pvalue2)
