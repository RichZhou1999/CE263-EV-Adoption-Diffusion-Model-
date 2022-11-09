import numpy as np
import pandas as pd

# data1 = pd.read_csv("WA_latest_snapshot_BEV.csv")
# data2 = pd.read_csv("WA_latest_snapshot_PHEV.csv")
# census = pd.read_csv("wa_census_data.csv")
#
# BEV = pd.merge(data1, census, on="zip code")
# BEV.to_csv('PHEV_data.csv')
# print(sum(BEV['Population']))
BEV = pd.read_csv("BEV_data.csv")
BEV = BEV.drop(["Unnamed: 0"], axis=1)
BEV.to_csv('BEV_data.csv', index=False)
# z1 = list(census['zip code'])
# z2 = list(data2['zip code'])
# z1.sort()
# z2.sort()
# z1 = set(z1)
# z2 = set(z2)
# n = 0
# for i in range(len(z2)):
#     if z1[i] == z2[i]:
#         n += 1
# print(print(len(z1 & z2)))