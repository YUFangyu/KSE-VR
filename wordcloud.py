import os
import joblib 
from gensim.corpora import MmCorpus, Dictionary

# work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'HalfLifeA_1')
work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'hlA_0721')
os.chdir(work_dir)


wordcloud_list = []

for i in range(3, 53):
    model = joblib.load('lda_train_HalfLifeA_{}.pkl'.format(i))
    wordcloud_list.append(model)
    
for line in wordcloud_list:
    print(line.show_topic(0,50))
    print('\n')


# with open('HalfLifeA_wordcloud.txt','a+',encoding='utf-8') as f_txt:
#     for line in wordcloud_list:
#         wc_str = ', '.join(line)
        
#         f_txt.write(wc_str)
#         f_txt.write('\n')
        
        
