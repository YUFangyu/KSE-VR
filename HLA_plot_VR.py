import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import joblib
from gensim.corpora import MmCorpus, Dictionary

import sys
import os
import numpy as np

work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'HalfLifeA')
os.chdir(work_dir)


for game in ['HalfLifeA']:
    def coherence(game):
        coherence_list = []

        for i in range(3, 101):
            coherence_list.append(joblib.load('coherence_value_'+ game + '_' + str(i) + '.pkl'))
        
        return coherence_list

x_axis = np.arange(3,101)

y_A = coherence("HalfLifeA")

# min_index_A = y_A.index(min(y_A))
# print("min topic numbers: {}".format(int(min_index_A)+3))

l_A = plt.plot(x_axis, y_A, label = 'Half-LifeA')

plt.grid(True)

# plt.title('Coherence_Half-LifeA')


plt.xlabel('Numbers of topic')
plt.ylabel('Coherence score')


plt.savefig('C:\\Users\\YU Yang\\Desktop\\coherence_halflife5.png')
plt.show()