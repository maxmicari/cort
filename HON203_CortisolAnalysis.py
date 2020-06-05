#HON203: Macaque and Human Cortisol Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
plt.close('all')
fgc_human = np.concatenate((np.random.randint(35,78,size=(280)),np.random.randint(35,78,size=(280))))
fgc_monkey = np.concatenate((np.random.randint(15,26,size=(280)),np.random.randint(5,22,size=(280))))
print('FGC Concentration is measured in nanograms/gram')
data = {'Participant': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
     'FGC_Concentration_Human': fgc_human,
     'FGC_Concentration_Monkey': fgc_monkey,
     'Day': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
             2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
             3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
             4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
             5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
             7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
             8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
             9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
             10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
             11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
             12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
             13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
             14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
             15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
             16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
             17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,
             18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,
             19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,
             20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
             21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,
             22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
             23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
             24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,
             25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,
             26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,
             27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,
             28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28],
     'Time': np.random.randint(0,24,size=(560))}

df = pd.DataFrame(data=data)
#standardize fgc data
fgcs = df[['FGC_Concentration_Human','FGC_Concentration_Monkey']].copy()
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(fgcs)
scaled_df = pd.DataFrame(scaled_df)
scaled_df = scaled_df.rename(columns={0: "FGC_Concentration_Human", 1: "FGC_Concentration_Monkey"})
scaled_df['Day'] = df['Day']

#correlation test on standardized data
r_val = scaled_df['FGC_Concentration_Monkey'].corr(scaled_df['FGC_Concentration_Human'], method='pearson')
r_val2 = df['FGC_Concentration_Monkey'].corr(df['FGC_Concentration_Human'], method='pearson')
coval = scaled_df['FGC_Concentration_Monkey'].cov(scaled_df['FGC_Concentration_Human'])
print(r_val2)
print(coval)

sns.regplot(x='Day', y='FGC_Concentration_Human', color='DarkBlue',data=scaled_df)
sns.regplot(x='Day', y='FGC_Concentration_Monkey', color='DarkRed',data=scaled_df)
plt.title('Misleading Fecal Glucocorticoid Concentration Over Time in Humans and Macaques with Best Fit Lines')
plt.xlabel('Day\nIn this scenario, human cortisol levels remain nominal after the onsen')
plt.ylabel('Standardized Fecal Glucocorticoid Concentrations')
plt.show()

ax = scaled_df.plot.scatter(x='Day', y='FGC_Concentration_Human', color='DarkBlue', label='Humans', s=20);
scaled_df.plot.scatter(x='Day', y='FGC_Concentration_Monkey', color='DarkRed', label='Macaques', s=20, ax=ax);
plt.title('Fecal Glucocorticoid Concentration Over Time in Humans and Macaques (STANDARDIZED)')
plt.xlabel('Day')
plt.ylabel('Fecal Glucocorticoid Concentration (ng/g)')
plt.show()

df['Monkey_Normal'] = df['FGC_Concentration_Monkey'].transform([np.log])
df['Human_Normal'] = df['FGC_Concentration_Human'].transform([np.log])
print(df['FGC_Concentration_Monkey'].describe())
print(df['FGC_Concentration_Human'].describe())

ax = df.plot.scatter(x='Day', y='Human_Normal', color='DarkBlue', label='Humans', s=20);
df.plot.scatter(x='Day', y='Monkey_Normal', color='DarkRed', label='Macaques', s=20, ax=ax);
plt.title('Fecal Glucocorticoid Concentration Over Time in Humans and Macaques (standardized)')
plt.xlabel('Day')
plt.ylabel('Fecal Glucocorticoid Concentration (ng/g)')
plt.show()
