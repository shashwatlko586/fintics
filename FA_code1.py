# install the required packages
!pip install yfinance
!pip install yahoofinancials


# display multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# import packages
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# load data
reliance_df = yf.download('RELIANCE.NS', 
                          start='2010-12-31',
                          progress=False)
reliance_df.sort_index(ascending=False, inplace=True)
nifty_df = yf.download('^NSEI', 
                          start='2010-12-31',
                          progress=False)
nifty_df.sort_index(ascending=False, inplace=True)


# view sample data
reliance_df.head()
nifty_df.head()


# plot adjusted close
matplotlib.rcParams["figure.dpi"] = 250
plt.plot(reliance_df['Adj Close'])



# describe the dataset for eda
reliance_df.isnull().sum().sum()
reliance_df.describe()

nifty_df.isnull().sum().sum()
nifty_df.describe()



# review rows were volume is zero
reliance_df = reliance_df[reliance_df["Volume"] > 0]
nifty_df = nifty_df[nifty_df["Volume"] > 0]



# view dataframe shape
reliance_df.shape
print('\n')
nifty_df.shape



# convert to weekly/monthly data
reliance_weekly_df = reliance_df.resample("1w").mean()
reliance_weekly_df.shape
reliance_monthly_df = reliance_df.resample("1m").mean()
reliance_monthly_df.shape



# compute log-return and add it as a column in the dataframe
reliance_df["Return"] = np.log(reliance_df["Adj Close"]) - np.log(reliance_df["Adj Close"].shift(1))
reliance_df.head()



reliance_df.info()
print('\n')
reliance_df.isnull().sum()
reliance_df = reliance_df.dropna()
print('\n')
reliance_df.isnull().sum()



# Tests of Normality

# Jarque–Bera Test
from scipy.stats import jarque_bera
statistic = jarque_bera(reliance_df["Return"])
print(statistic)

# Anderson-Darling Normality Test
from scipy.stats import anderson
statistic = anderson(reliance_df["Return"])
print(statistic)

# Kolmogorov-Smirnov Test
from scipy.stats import kstest
statistic = kstest(reliance_df["Return"], 'norm')
print(statistic)




# Tests of Stationarity

# Augmented Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller
statistic = adfuller(reliance_df["Return"])
print(statistic)

# Ljung-Box Test
from statsmodels.stats.diagnostic import acorr_ljungbox
statistic = acorr_ljungbox(reliance_df["Return"], lags=[1])
print(statistic)



