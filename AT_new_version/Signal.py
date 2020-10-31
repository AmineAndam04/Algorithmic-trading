import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *
from .utils import adjustsignal 

def signal_smm(df,n):
	"""
	Trading signals of rule1 of simple moving average 
	For more information about rule1 see documentation 
		Inputs: 
				Input | Type                             | Description
				=================================================================================
				 df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
				 n    | int                              | Period
		Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	    **Signals:   1----> Buy
	                 0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	signal['signal']=0
	signal["compa"]=0
	MMS=smm(df,n)["SMM"]
	signal["compa"].loc[n:]=np.where(df[n:] > MMS[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_smm2(df,ws,wl):
	"""
	Trading signals of rule2 of simple moving average 
	For more information about rule2 see documentation 	 
	Inputs: 
			Input   | Type                             | Description
			=================================================================================
			 df     |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         ws     |int                               | The period of the shorter moving average
	         wl     |int                               | The periode if the longer moving average
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell

	"""
	signal=pd.DataFrame(index=df.index)
	signal['compa']=np.nan
	signal["MMS_COURT"]=smm(df,ws)["SMM"]
	signal["MMS_LONG"]=smm(df,wl)["SMM"]
	signal["compa"].loc[wl:]=np.where(signal["MMS_COURT"][wl:] > signal["MMS_LONG"][wl:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_macd1(df,ws,wl):
	"""
	Trading signals using rule1 of Moving Average Convergence Divegence
	For more information about rule1 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         ws    |int                               | The period of the shorter moving average
	         wl    |int                               | The periode if the longer moving average
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""

	signal=pd.DataFrame(index=df.index)
	MACD=macd(df,ws,wl)["MACD"]
	signal["compa"]=np.nan
	signal["compa"][wl:]=np.where(MACD[wl:]>0 ,1 ,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_macd2(df,ws,wl,wsig=9):
	"""
	Trading signals using rule2 the Moving Average Convergence Divegence
	For more information about rule2 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         ws    |int                               | The period of the shorter moving average
	         wl    |int                               | The periode if the longer moving average
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	MACD=macd(df,ws,wl,wsig)[["MACD","MACDsignal"]]
	signal["compa"][wl:]=np.where(MACD["MACD"][wl:] > MACD["MACDsignal"][wl:] ,1 ,0 )
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def  signal_rsi(df,n):
	"""
	Trading signals using  Relative Strength index (RSI)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
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
	return sig


def signal_bollinger(df,w,k):
	"""
	trading signals using  the Bollinger Bands
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | Periods of moving average
	         k     |int                               | The number of stradard diviations
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
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
	return sig


def signal_momentum(df,w,wsig=9):
	"""
	Trading signals using the momentum
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | The period 
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	MOM=momentum(df,w,wsig=9)[["MOM","MOMsignal"]]
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where(MOM["MOM"][w:] > MOM["MOMsignal"][w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_stochastic1(df,high,low ,n , w):
	"""
	Trading signals using rule1 the Stochastic oscillator : %K and %D
	For more information about rule1 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |int                               | %D periods 
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
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
	return sig


def signal_stochastic2(df,high,low ,n , w):
	"""
	Trading signals using rule2 the Stochastic oscillator : %K and %D
	For more information about rule2 see documentation 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |int                               | %D periods 
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	k=stochastic(df ,high,low, n , w)[["%K","%D"]]
	signal["compa"]=np.nan
	signal["compa"][n+w :]=np.where(k["%K"][n+w:] > k["%D"][n+w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_obv(df,vol,n):
	"""
	Trading signals of  the On Balance Volume (OBV) & the signal line of OBV
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame                  | Prices
	         vol  |pandas.DataFrame                  | Volumes
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	ob=obv(df,vol)["OBV"]
	obs=pd.Series(ob.rolling(n).mean(), name='obvsignal')
	signal["compa"]=np.nan 
	signal["compa"][n:]=np.where( ob[n:]  > obs[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_roc(df,w):
	"""
	Trading signals using the Rate of change (ROC)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         w    | int                              | Period
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	roc=rate_of_change(df,w)["ROC"]
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where( roc[w:]> 0,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return  signal["signal"]


def signal_mfi(df,vol,high,low,n):
	"""
	Trading signals using  the Money Flow Index (MFI)
	Inputs: 
	        Input   | Type                             | Description
	       =========================================================================================
	         close  |pandas.DataFrame or pandas.Series | Prices
	         volume |pandas.DataFrame or pandas.Series | Volumes
	         High   |pandas.DataFrame or pandas.Series | Highest high 
	         low    |pandas.DataFrame or pandas.Series | Lowest low
	         n      |int                               | Periods 
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell

	"""


	signal=pd.DataFrame(index=df.index)
	mfi=MFI(df,vol,high,low,n)['MFI']
	signal["compa_achat"]=np.nan
	signal["compa_vente"]=np.nan
	signal["compa_achat"][n:]=np.where(mfi[n:] > 30,1,0)
	signal["signal_achat"]=signal["compa_achat"].diff()
	signal["compa_vente"][n:]=np.where(mfi[n:] < 70,4,2)
	signal["signal_vente"]=signal["compa_vente"].diff()
	sig=np.where(signal["signal_achat"]==1,1,0)+np.where(signal["signal_vente"]==2,-1,0)
	sig=adjustsignal(sig)
	sig=pd.Series(sig, index=df.index)
	return sig


def signal_cho(df,vol,high,low,n,ws,wl):
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
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	ch=cho(df,vol,high,low,n,ws,wl)["CHO"]
	signal["compa"][ws+wl:]=np.where(ch[ws+wl:] > 0, 1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_nvi(df,vol,n):
	"""
	Trading signals using the Negative Volume Index (NVI) & the signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	nv=nvi(df,vol)
	nvis=pd.Series(nv.rolling(n).mean(), name="NVIsignal")
	signal["compa"][n:]=np.where(nv[n:] > nvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


def signal_pvi(df,vol,n):
	"""
	Trading signals using the Positive Volume Index (PVI) & the signal line 
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
				Output | Type                             | Description
				================================================================================= 
				       | pandas.Series                    | Signals**
	**Signals:   1----> Buy
	             0----> Sell
	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	pv=pvi(df,vol)
	pvis=pd.Series(pv.rolling(n).mean(), name="PVIsignal")
	signal["compa"][n:]=np.where(pv[n:] > pvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]


