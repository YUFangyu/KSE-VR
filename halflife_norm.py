import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
# from gensim.corpora import MmCorpus, Dictionary
# from wordcloud import WordCloud
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()   

work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
os.chdir(work_dir)


df = pd.read_excel("HalfLife_bar.xlsx", 'Sheet3', index_col=0)


fig,axes=plt.subplots(nrows=1, ncols=2)

sns.distplot(df['HLA (p=0.327)'], ax=axes[0]) #左图
sns.distplot(df['HL2 (p=0.233)'], ax=axes[1]) #右图
axes[1].set_ylabel("")
plt.savefig("halflife_norm3")

plt.show()


