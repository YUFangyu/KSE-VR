import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import joblib
from gensim.corpora import MmCorpus, Dictionary
# from wordcloud import WordCloud
import sys
import os
import numpy as np

work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'lda_vr2')
os.chdir(work_dir)

for game in ['HalfLife2', 'HalfLifeA']:
# for game in ['HalfLifeA']:
    def coherence(game):
        coherence_list = []

        for i in range(3, 101):
            coherence_list.append(joblib.load('coherence_value_'+ game + '_' + str(i) + '.pkl'))
            # print(coherence_list)
        return coherence_list

x_axis = np.arange(3,101)

y_2 = coherence("HalfLife2")
y_A = coherence("HalfLifeA")


min_index_2 = y_2.index(min(y_2))
min_index_A = y_A.index(min(y_A))
print(min_index_2, min_index_A)


l_2 = plt.plot(x_axis, y_2, label = 'HalfLife2')
l_A = plt.plot(x_axis, y_A, label = 'HalfLifeA')


# plt.legend(loc='best')
plt.legend(loc='lower right')
# 'lower right'

# df.plot(legend=False)



plt.grid(True)
plt.title('Coherence_HalfLife2/A')
# plt.title('Coherence_HalfLifeA')


plt.xlabel('Numbers of topic')
plt.ylabel('Coherence value')


# plt.savefig('C:\\Users\\YU Yang\\Desktop\\coherence_halflife3.png')
plt.show()