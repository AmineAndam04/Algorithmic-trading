import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *


 
def plot_smm(df,n):
	"""
	Plots the simple moving average 
	Inputs: 
			Input | Type                             | Description
			=================================================================================
			 df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period 

	"""
	fig = plt.figure()
	ax1 = fig.add_subplot(111, ylabel='Values')
	MMS=smm(df,n)
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	MMS['SMM'].plot(ax=ax1, color='r', lw=2., legend=True)
	plt.legend(loc=0)
	plt.show()

def plot_emm(df,n):
    """
    Plots the exponential moving average
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Values')
    MME=emm(df,n)
    df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
    MME['EMM'].plot(ax=ax1, color='r', lw=2., legend=True)
    plt.legend(loc=0)
    plt.show()

def plot_macd(df,ws,wl, wsig=9):
	"""
	Plots the Moving Average Convergence Divegence
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         ws    |int                               | The period of the shorter moving average
	         wl    |int                               | The periode if the longer moving average
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9

	"""
	MACD=macd(df,ws,wl,wsig)
	MACD['hist']=MACD["MACD"]-MACD['MACDsignal']
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD['MACD'].plot(ax=ax2, color='black', lw=2., legend=True)
	MACD["MACDsignal"].plot(ax=ax2, color='g', lw=2., legend=True)
	ax3 = fig.add_subplot(313, ylabel='MACD')
	MACD["hist"].plot(ax=ax3, color='r', linestyle="--",grid=True, legend=True,use_index=False)
	plt.show()

def plot_rsi(df,n):
    """
    Plots the Relative Strength index (RSI)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period

    """
    rs=rsi(df,n)["RSI"]*100
    fig = plt.figure()
    ax1 = fig.add_subplot(311, ylabel='Values')
    df.plot(ax=ax1, color='black', lw=2., legend=True,figsize=(13,9))
    ax2=fig.add_subplot(312, ylabel='RSI')
    rs.plot(ax=ax2, color='b', lw=2., legend=True)
    plt.show()

def plot_bollinger(df,w,k):
	"""
	Plots the Bollinger Bands
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | Periods of moving average
	         k     |int                               | The number of stradard diviations

	"""
	BB=bollinger(df,w,k)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	BB[["Input","BBUP","BBDOWN"]].plot(ax=ax1,figsize=(13,9))
	ax1.fill_between(df.index,BB["BBUP"],BB["BBDOWN"],facecolor='red', alpha=0.4)


def plot_momentum(df,w,wsig=9):
	"""
	Momentum
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | The period 
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9

	"""
	MOM=momentum(df,w,wsig)[['MOM','MOMsignal']]
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='Momentum ')
	MOM["MOM"].plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	MOM["MOMsignal"].plot(ax=ax2, color='r', lw=2., legend=True)
	plt.show()

def plot_roc(df,w):
	"""
	Plots the Rate of change (ROC)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         w    | int                              | Period

	"""
	roc=rate_of_change(df,w)['ROC']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='ROC')
	roc.plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	plt.show()

def plot_stochastic(df ,high,low, n , w):
	"""
	Plots the Stochastic oscillator : %K and %D
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |                                  | %D periods 

	"""
	stoch=stochastic(df ,high,low, n , w)[['%K','%D']]
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(312, ylabel='%K')
	stoch['%K'].plot(ax=ax2, color='b', lw=2., legend=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3 = fig.add_subplot(313, ylabel='%K ET %D')
	stoch.plot(ax=ax3, lw=2., legend=True)
	plt.show()

def plot_obv(df,vol):
	"""
	Plots the On Balance Volume (OBV) & the signal line of OBV
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame                  | Prices
	         vol  |pandas.DataFrame                  | Volumes

	"""
	ob=obv(df,vol)['OBV']
	obvsignal=pd.Series(ob.rolling(10).mean(),name= 'obvsignal')
	ob.plot(legend=True, figsize=(10,5))
	obvsignal.plot(legend=True)

def plot_williams(df,n):
	"""
	Plots the Williams %R
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Periods

	"""
	wil=williams(df,n)['%R']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='Williams %R')
	wil.plot(ax=ax2, color='b', lw=2., legend=True)
	plt.axhline(-20,color="green")
	plt.axhline(-80,color="red")
	plt.show()

def plot_MFI(close,volume,high,low,n):
	"""
	Plots the Money Flow Index (MFI)
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods 

	"""
	mfi=MFI(close,volume,high,low,n)['MFI']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='MFI')
	mfi.plot(ax=ax2, color='b', lw=2., legend=True)
	plt.show()

def plot_cho(close,volume,high,low,n,ws,wl):
	"""
	Plots the Chaikin Oscillator
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods
	         ws     |int                               | The period of the shorter moving average
	         wl     |int                               | The periode if the longer moving average

	"""
	ch=cho(close,volume,high,low,n,ws,wl)['CHO']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='CHO')
	ch.plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	plt.show()


def plot_nvi(close,volume):
	"""
	Plots the Negative Volume Index (NVI) & the signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes

	"""
	nv=nvi(close,volume)
	nvisignal=pd.Series(nv.rolling(10).mean(), name= "NVIsignal")
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='NVI')
	nv.plot(ax=ax2, lw=2., legend=True,grid=True)
	nvisignal.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.show()

def plot_pvi(close,volume):
	"""
	Plots the Positive volume index (PVI)  & thz signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes

	"""
	pv=pvi(close,volume)
	pvisignal=pd.Series(pv.rolling(10).mean(), name= "PVIsignal")
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='PVI')
	pv.plot(ax=ax2, lw=2., legend=True,grid=True)
	pvisignal.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.show()
