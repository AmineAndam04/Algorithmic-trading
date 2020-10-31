import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *
from .utils import adjustsignal
from .Performance import pmv

def plt_smm1(df,n):
    """
     Plots the rule1 using the simple moving average 
     Go to documentation to know the different rules of SMM that we are using in our package 
     Inputs: 
    		Input | Type                             | Description
    		=================================================================================
    		 df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
             n    | int                              | Period 

    """
    signal=pd.DataFrame(index=df.index)
    signal['signal']=0
    signal["compa"]=0
    MMS=smm(df,n)["SMM"]
    signal["compa"].loc[n:]=np.where(df[n:] > MMS[n:],1,0)
    signal["signal"]=adjustsignal(signal["compa"].diff())
    pmval=pmv(df,signal["signal"])
    pmval=pd.Series(pmval,index=df.index)
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='Values')
    df.plot(ax=ax1, color='g', lw=.5)
    MMS.plot(ax=ax1, lw=.5, figsize=(13,9))
    ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
    ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
    plt.legend(["Price","SMM","Buy","Sell"])  
    ax2 = fig.add_subplot(212, ylabel='Capital gain/loss')
    plt.title("Simple Moving Average Trading Strategy")
    ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
    ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
    plt.legend(["Plus_value","Moins_value"])

def plt_smm2(df,ws,wl):
	"""
	Plots the rule2 using the simple moving average 
	 Go to documentation to know the different rules of SMM that we are using in our package 
	Inputs: 
			Input   | Type                             | Description
			=================================================================================
			 df     |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         ws     |int                               | The period of the shorter moving average
	         wl     |int                               | The periode if the longer moving average

	"""
	signal=pd.DataFrame(index=df.index)
	signal['compa']=np.nan
	signal["MMS_COURT"]=smm(df,ws)["SMM"]
	signal["MMS_LONG"]=smm(df,wl)["SMM"]
	signal["compa"].loc[wl:]=np.where(signal["MMS_COURT"][wl:] > signal["MMS_LONG"][wl:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	signal["MMS_COURT"].plot(ax=ax1, lw=2.)
	signal["MMS_LONG"].plot(ax=ax1, lw=2.)
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["Price","SMM_short","SMM_long","Buy","Sell"])
	plt.title("Double Moving Average Trading Strategy")
	ax2 = fig.add_subplot(212, ylabel='Capital gain/loss')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_macd1(df,ws,wl):
	"""
	Plots trading signals using rule1 of Moving Average Convergence Divegence
	For more information about rule1 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         ws    |int                               | The period of the shorter moving average
	         wl    |int                               | The periode if the longer moving average
	"""
	signal=pd.DataFrame(index=df.index)
	MACD=macd(df,ws,wl)["MACD"]
	signal["compa"]=np.nan
	signal["compa"][wl:]=np.where(MACD[wl:]>0 ,1 ,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.title("MACD Strategy")
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD.plot(ax=ax2, color='black', lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])



def plt_macd2(df,ws,wl,wsig=9):
	"""
	Plots trading signals using rule2 the Moving Average Convergence Divegence
	For more information about rule2 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         ws    |int                               | The period of the shorter moving average
	         wl    |int                               | The periode if the longer moving average
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9


	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	MACD=macd(df,ws,wl,wsig)[["MACD","MACDsignal"]]
	signal["compa"][wl:]=np.where(MACD["MACD"][wl:] > MACD["MACDsignal"][wl:] ,1 ,0 )
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.title("MACD Strategy")
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def  plt_rsi(df,n):
	"""
	Plots trading signals using  Relative Strength index (RSI)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period


	"""
	signal=pd.DataFrame(index=df.index)
	rs=rsi(df,n)["RSI"]
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(rs[n:] > 0.3,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(rs[n:] < 0.7,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["signal"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal ==-1], 'v', markersize=7, color='r')
	plt.title("RSI Strategy")
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='RSI')
	rs.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(0.7,color="green")
	plt.axhline(0.3,color="red")
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_bollinger(df,w,k):
	"""
	Plots trading signals using  the Bollinger Bands
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | Periods of moving average
	         k     |int                               | The number of stradard diviations


	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa2"]=np.nan
	bb=bollinger(df,w,k)[["BBDOWN","BBUP"]]
	signal["compa"][w:] = np.where( (df[w:] > bb["BBUP"][w:] ) ,1,0)
	signal["compa2"][w:] = np.where( (df[w:] < bb["BBDOWN"][w:] ) ,4,2)
	signal["signal"]=signal["compa"].diff()
	signal["signal2"]=signal["compa2"].diff()
	sig=np.where(signal["signal"]==1,-1,0)+np.where(signal["signal2"]==2,1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["sig"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.fill_between(df.index,bb["BBUP"],bb["BBDOWN"],facecolor='red', alpha=0.2)
	bb.plot(ax=ax1, lw=.5)
	ax1.plot(signal.loc[signal.sig==1].index ,df[signal.sig==1],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.sig==-1].index ,df[signal.sig==-1],'v', markersize=7, color='r')
	plt.legend(["Values","BBDOWN","BBUP","Buy",  "Sell"])
	plt.title("Bondes de Bollinger Trading Strategy")
	ax2=fig.add_subplot(212, ylabel='Capital gain/loss')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_momentum(df,w,wsig=9):
	"""
	Plots trading signals using the momentum
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | The period 
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9

	"""
	MOM=momentum(df,w,wsig=9)[["MOM","MOMsignal"]]
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where(MOM["MOM"][w:] > MOM["MOMsignal"][w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["Values","Buy","Sell"])
	ax2=fig.add_subplot(312, ylabel='MOM')
	MOM.plot(ax=ax2, legend=True, grid=True)
	plt.title("MOM Strategy")
	ax3=fig.add_subplot(313, ylabel='Capital gain/sell')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_stochastic1(df,high,low ,n , w):
	"""
	Plots trading signals using rule1 the Stochastic oscillator : %K and %D
	For more information about rule1 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |int                               | %D periods 


	"""
	signal=pd.DataFrame(index=df.index)
	k=stochastic(df ,high,low, n , w)["%K"]
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(k[n:] > 20,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(k[n:] < 80,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	signal["signal"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Stochastic')
	k.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_stochastic2(df,high,low,n , w):
	"""
	Plots trading signals using rule2 the Stochastic oscillator : %K and %D
	For more information about rule2 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |int                               | %D periods 


	"""
	signal=pd.DataFrame(index=df.index)
	k=stochastic(df ,high,low, n , w)[["%K","%D"]]
	signal["compa"]=np.nan
	signal["compa"][n+w :]=np.where(k["%K"][n+w:] > k["%D"][n+w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Stochastic')
	k.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_obv(df,vol,n):
	"""
	Plots trading signals of  the On Balance Volume (OBV) & the signal line of OBV
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame                  | Prices
	         vol  |pandas.DataFrame                  | Volumes

	"""
	signal=pd.DataFrame(index=df.index)
	ob=obv(df,vol)["OBV"]
	obs=pd.Series(ob.rolling(n).mean(), name='obvsignal')
	signal["compa"]=np.nan 
	signal["compa"][n:]=np.where( ob[n:]  > obs[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='ON-Banlance Volume')
	ob.plot(ax=ax2, lw=2., legend=True,grid=True)
	obs.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_roc(df,w):
	"""
	Plots  trading signals using the Rate of change (ROC)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         w    | int                              | Period


	"""
	signal=pd.DataFrame(index=df.index)
	roc=rate_of_change(df,w)["ROC"]
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where( roc[w:]> 0,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Rate of change')
	roc.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_mfi(close,volume,high,low,n):
	"""
	Plots trading signals using  the Money Flow Index (MFI)
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods 

	"""
	signal=pd.DataFrame(index=close.index)
	mfi=MFI(close,volume,high,low,n)['MFI']
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(mfi[n:] > 80,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(mfi[n:] < 20,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=close.index)
	signal["signal"]=sig
	pmval=pmv(close,sig)
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	close.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["Values","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='MFI')
	mfi.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_cho(close,volume,high,low,n,ws,wl):
	"""
	Plots trading signals using the Chaikin Oscillator
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
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	ch=cho(close,volume,high,low,n,ws,wl)["CHO"]
	signal["compa"][ws+wl:]=np.where(ch[ws+wl:] > 0, 1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["Values","CHO","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Chaikin Oscillator')
	ch.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_nvi(close,volume,n):
	"""
		Plots trading signals using the Negative Volume Index (NVI) & the signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes

	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	nv=nvi(close,volume)
	nvis=pd.Series(nv.rolling(n).mean(), name="NVIsignal")
	signal["compa"][n:]=np.where(nv[n:] > nvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["Values","NVI","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Negative Volume Index')
	nv.plot(ax=ax2,lw=2., legend=True, grid=True)
	nvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/sell')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_pvi(close,volume,n):
	"""
	Plots trading signals using the Positive Volume Index (PVI) & the signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes

	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	pv=pvi(close,volume)
	pvis=pd.Series(pv.rolling(n).mean(), name="PVIsignal")
	signal["compa"][n:]=np.where(pv[n:] > pvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='Values')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["Values","NVI","Buy","Sell"])
	ax2 = fig.add_subplot(312, ylabel='Positive Volume Index')
	pv.plot(ax=ax2,lw=2., legend=True, grid=True)
	pvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='Capital gain/loss')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])
