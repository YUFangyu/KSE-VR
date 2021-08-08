import pandas as pd
import datetime as dt
import os

work_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
os.chdir(work_dir)

game = 'HalfLife2'

df = pd.read_csv("{}_csv.csv".format(game), engine='c', index_col=False)

df1 = df.drop(columns=['author_id', 'funny','length'])
df1['updated_time'] = pd.to_datetime(df1['updated_time']).dt.date

df1.loc[df1['attitude'] =='Not Recommended', 'attitude'] = 0
df1.loc[df1['attitude'] =='Recommended', 'attitude'] = 1

df1['updated_time']= pd.to_datetime(df1['updated_time']).dt.to_period('M')


df2 = df1
df2.to_csv("{}_simple_month.csv".format(game), index=0)