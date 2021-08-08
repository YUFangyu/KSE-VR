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

df = pd.read_clipboard()

work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
os.chdir(work_dir)


df = pd.read_excel("HalfLife_bar.xlsx", 'Sheet3', index_col=0)


ax = df.plot(kind='bar', figsize= (12,8),grid=True, rot=1)
# ax = df.plot(kind='bar',title = 'Topics_comparison',grid=True, rot=1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel("Topics")
ax.set_ylabel("Positive rates(%)")
plt.legend(loc='lower right')
plt.savefig('halflife_bar16.png')
plt.show()