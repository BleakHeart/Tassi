"""
Guardare la lezione del 14 aprile 2021 in caso di approfondimenti
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime


Path = 'Data/'

stock = pd.read_csv(Path + 'LBMA-GOLD.csv')

ts = stock[['Date', 'USD (AM)', 'USD (PM)']].dropna()
ts['Date'] = ts[['Date']].astype('datetime64[ns]')
rate = ts[['USD (AM)', 'USD (PM)']].mean(axis=1).diff() / ts[['USD (AM)', 'USD (PM)']].mean(axis=1)
rate.iloc[0] = 0

'''
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ts[['Date']], ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1), c='k')
ax1.set_ylabel(r'S(t)')
ax1.grid()

ax2 = fig.add_subplot(gs[1, 0])
print(rate.iloc[1:].mean())
print(ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1).mean(), ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1)[0])
r_m = (ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1).mean() - ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1)[0]) / ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1)[0]
ax2.plot(ts[['Date']], rate, c='k', label=r'$\bar{r}$=%0.3f' %(r_m))
ax2.set_xlabel('Date')
ax2.set_ylabel(r'r(t)')
ax2.legend()
ax2.grid()

plt.show()'''

symbols = {'GCM21', 'GCQ21', 'GCV21', 'GCZ21'}  # 'GCJ21', 'GCK21'

for symbol in symbols:
    globals()[f'{symbol}_df'] = pdr.get_data_yahoo(symbols=f'{symbol}.CMX', start=datetime(2020, 4, 13), end=datetime(2021, 4, 9))
    globals()[f'{symbol}_df'].reset_index(inplace=True, drop=False)

'''for symbol in symbols:
    print(globals()[f'{symbol}_df'])'''

'''for symbol in symbols:
    print(symbol, sum(ts['Date'].isin(globals()[f'{symbol}_df']['Date'])), len(globals()[f'{symbol}_df']), len(ts))'''


Spot_Fut_df = ts.loc[ts['Date'].isin(GCM21_df['Date'])].copy(deep=True)
Spot_Fut_df['USD'] = Spot_Fut_df[['USD (AM)', 'USD (PM)']].mean(axis=1)
Spot_Fut_df.drop(columns=['USD (AM)', 'USD (PM)'], inplace=True)

nomen = ['Jun21', 'Aug21', 'Oct21', 'Dec21']

date = ['28/06/2021', '27/08/2021', '27/10/2021', '29/12/2021']

dic1 = dict(zip(symbols, nomen))

for symbol in symbols:
    Spot_Fut_df[f'{dic1[symbol]}_Fut'] = Spot_Fut_df['Date'].map(globals()[f'{symbol}_df'].set_index('Date')['Adj Close'])

print(Spot_Fut_df.head())

Basis_Spot = Spot_Fut_df[['Date', 'USD']]

for symbol in symbols:
    Basis_Spot[f'{dic1[symbol]}_Bs'] = Spot_Fut_df[f'{dic1[symbol]}_Fut'] - Spot_Fut_df['USD']

Basis_Spot.rename(columns={'USD': 'Spot'}, inplace=True)


RR_df = Basis_Spot[['Date', 'Spot']]

for symbol in symbols:
    RR_df[f'{dic1[symbol]}_RR'] = Spot_Fut_df[f'{dic1[symbol]}_Fut'] / Spot_Fut_df['USD'] - 1.

print(RR_df.head())


date = ['28/06/2021', '27/08/2021', '27/10/2021', '29/12/2021']
End_date = [pd.to_datetime(x, format='%d/%m/%Y') for x in date]

Date_dict = dict(zip(symbols, End_date))

for symbol in symbols:
    globals()[f'{dic1[symbol]}_YtM'] = ((Date_dict[symbol] - RR_df.loc[:, 'Date']) / pd.to_timedelta(365, unit='D')).values

ARR_df = Basis_Spot[['Date', 'Spot']]

for symbol in symbols:
    ARR_df[f'{dic1[symbol]}_ARR'] = np.power(RR_df[f'{dic1[symbol]}_RR'] + 1, 1 / globals()[f'{dic1[symbol]}_YtM'] - 1) - 1

ARR_df.drop(columns='Spot', inplace=True)

for symbol in symbols:
    plt.plot(RR_df['Date'], RR_df[f'{dic1[symbol]}_RR'], label=f'RR {dic1[symbol]}')

plt.ylabel('Annual Recurring Revenue')
plt.xlabel('Time')
plt.grid()
plt.legend()
plt.show()


for symbol in symbols:
    plt.plot(ARR_df['Date'], ARR_df[f'{dic1[symbol]}_ARR'], label=f'ARR {dic1[symbol]}')

plt.ylabel('Annual Recurring Revenue')
plt.xlabel('Time')
plt.grid()
plt.legend()
plt.show()

'''#print(ts)
#plt.plot(rate)
plt.plot(ts[['Date']], ts[['EURO (AM)', 'EURO (PM)']].mean(axis=1), c='k')
#plt.xlim(len(ts), 0)
plt.xlabel('Date')
plt.title('Gold Price (Eur)')
plt.show()'''