import pandas as pd
import numpy as np 


def smm(df,n):
	"""
	Simple Moving Average
	Inputs: 
			Input | Type                             | Description
			=================================================================================
			 df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ================================================================================= 
	               | pandas.DataFrame (2 columns)     | 1st column : the input df
	               |                                  | 2nd column(SMM): values of Moving average

	"""
	MMS = pd.Series(df.rolling(n, min_periods=n).mean(), name='SMM')
	df=pd.DataFrame(df)
	df=df.join(MMS)
	return df


def emm(df,n):
	"""
	Exponential Moving Average
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input df
	               |                                  | 2nd column(EMM): values of Moving average             

	"""
	exp=[df[:n].mean()]
	lamda=2/(1+n)
	for i in range(1,len(df)-n+1):
		val=(1-lamda)*exp[i-1]+lamda*df[i+n-1]
		exp.append(val)
	MME=pd.Series(index=df.index)
	MME[n-1:]=exp 
	MME.name="EMM"
	return MME

	
def macd(df,ws,wl, wsig=9):
	"""
	Moving Average Convergence Divegence
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
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column(MACD) : values of MACD
	               |                                  | 2nd column(MACDsignal): values of signal line 
	"""
	MMECOURT = pd.Series(df.ewm(span=ws, min_periods=ws,adjust=False).mean())
	MMELONG = pd.Series(df.ewm(span=wl, min_periods=wl,adjust=False).mean())
	MACD = pd.Series(MMECOURT - MMELONG, name='MACD' )
	MACDsign = pd.Series(MACD.ewm(wsig, min_periods=wsig).mean(), name='MACDsignal')
	MACD=pd.DataFrame(MACD)
	MACD = MACD.join(MACDsign)
	return MACD

def rsi(df,n):
 	"""
 	 Relative Strength index (RSI)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(RSI): values of RSI
	               |                                  |              

 	"""
 	diff=df.diff(1)
 	t=[]
 	for i in diff.values :
 		if i > 0:
 			t.append(i)
 		else :
 			t.append(0)
 	pos=pd.DataFrame(t,index=df.index)
 	diff=np.abs(pd.DataFrame(diff))
 	RSI=pos.rolling(n,min_periods=n).sum()/np.array((diff.rolling(n,min_periods=n).sum()))
 	df=pd.DataFrame(df)
 	df=df.join(RSI)
 	df.columns=["Input","RSI"] 
 	return df

def bollinger(df,w,k):
 	"""
 	Bollinger Bands 
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | Periods of moving average
	         k     |int                               | The number of stradard diviations
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (4 columns)     | 1st column: The input
	               |                                  | 2nd column(BBDOWN): Lower band
	               |                                  | 3rd column(BBUP) : Upper band  
	               |                                  | 4th column(BBMID): Middel band
 	"""
 	BBMID=df.rolling(w, min_periods=w).mean()
 	sigma=df.rolling(w, min_periods=w).std()
 	BBUP=BBMID + k*sigma
 	BBDOWN=BBMID - k*sigma
 	BBDOWN=pd.Series(BBDOWN,name='BBDOWN')
 	BBMID=pd.Series(BBMID,name='BBMID')
 	BBUP=pd.Series(BBUP,name='BBUP')
 	df=pd.DataFrame(df)
 	df=df.join(BBDOWN)
 	df=df.join(BBMID)
 	df=df.join(BBUP)
 	df.columns=['Input',"BBDOWN","BBMID","BBUP"]
 	return df
def momentum(df,w,wsig=9):
	"""
	Momentum
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         w     |int                               | The period 
	         wsig* |int                               | The period of the signal line 
	    * By default wsig= 9
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (3 columns)     | 1st column: The input
	               |                                  | 2nd column(MOM): Momentums
	               |                                  | 3rd column(MOMsignal) : The signal line 
	"""
	MOM=pd.Series(df.diff(w),name="MOM")
	MOMsignal=pd.Series(MOM.rolling(wsig, min_periods=wsig).mean(), name= "MOMsignal")
	df=pd.DataFrame(df)
	df=df.join(MOM)
	df=df.join(MOMsignal)
	return df

def rate_of_change(df,w):
	"""
	 Rate of change (ROC)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         w    | int                              | Period
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(ROC): values of ROC
	               |                                  |              

	"""
	ROC=pd.Series(100*(df.diff(w))/df.shift(w), name='ROC')
	df=pd.DataFrame(df)
	df=df.join(ROC)
	return df

def stochastic(df ,high,low, n , w):
	"""
	Stochastic oscillator : %K and %D
	Inputs: 
	        Input  | Type                             | Description
	       =========================================================================================
	         df    |pandas.DataFrame                  | Prices, volumes .....
	         high  |int                               | Highest high 
	         low   |int                               | Lowest low
	         n     |int                               | %K periods
	         w     |                                  | %D periods 
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (3 columns)     | 1st column : the input
	               |                                  | 2nd column(%K): values of %K
	               |                                  | 3rd column(%D): values of %D
	"""
	PH=high.rolling(n, min_periods=n).max()
	PB=low.rolling(n, min_periods=n).min()
	K=pd.Series(100*(df-PB)/(PH-PB), name='%K')
	D=pd.Series(K.rolling(w).mean(), name='%D')
	df=pd.DataFrame(df)
	df=df.join(K)
	df=df.join(D)
	return df 

def  obv(df,vol):
	"""
	On Balance Volume (OBV)
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame                  | Prices
	         vol  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(OBV): values of OBV

	"""
	prix=df.diff(1)/np.abs(df.diff(1))
	vec= vol*prix
	vec.iloc[0]=vol.iloc[0]
	OBV=pd.Series(vec.cumsum(), name= 'OBV')
	df=pd.DataFrame(df)
	df=df.join(OBV)
	return df

def williams(df,n):
	"""
	Williams %R
	Inputs: 
	        Input | Type                             | Description
	       =========================================================================================
	         df   |pandas.DataFrame or pandas.Series | Prices, volumes .....
	         n    | int                              | Periods
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input 
	               |                                  | 2nd column(%R): values of %R
	               |                                  |              
	"""
	PH=df.rolling(n, min_periods=n).max()
	PB=df.rolling(n, min_periods=n).min()
	R=pd.Series(-100*(PH-df)/(PH-PB), name='%R')
	df=pd.DataFrame(df)
	df=df.join(R)
	return df

def MFI(close,volume,high,low,n):
	"""
	 Money Flow Index (MFI)
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
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input (only prices)
	               |                                  | 2nd column(MFI): values of MFI
	"""
	ptyp=(close+high+low)/3
	PMF=[0]
	for i in range(1,len(ptyp)):
		if ptyp[i] > ptyp[i-1]:
			PMF.append(ptyp[i]*volume[i])
		else:
			PMF.append(0)
	PMF=pd.Series(PMF, name=' PMF',index=ptyp.index)
	MF=ptyp*volume
	ratio=pd.Series(100*PMF/MF)
	MFI=pd.Series(ratio.rolling(n,min_periods=n).mean(), name="MFI")
	df=pd.DataFrame(close)
	df=df.join(MFI)
	return df

def cho(close,volume,high,low,n,ws,wl):
	"""
	Chaikin Oscillator
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
	       ========================================================================================
	               | pandas.DataFrame (2 columns)     | 1st column : the input (only prices)
	               |                                  | 2nd column(CHO): values of CHO
	"""
	N=(2*close-low-high)/(high-low)
	adl=N*volume
	ADL=pd.Series(adl.rolling(n,min_periods=n).sum(), name='ADL')
	CHOL=pd.Series(ADL.ewm(ws,min_periods=ws).mean(),name='CHOL')
	CHOH=pd.Series(ADL.ewm(wl,min_periods=wl).mean(),name='CHOH')
	CHO=pd.Series(CHOL-CHOH, name="CHO")
	df=pd.DataFrame(close)
	df=df.join(CHO)
	return df

def nvi(close,volume):
	"""
	Negative Volume Index (NVI)
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.Series                    | NVI 
	"""
	roc=pd.Series(close.diff(1)/close.shift(1), name="ROC")
	nv=[np.nan,roc.iloc[1]]
	for i in range(2,len(volume)):
		if volume[i]< volume[i-1]:
			nv.append(nv[i-1]*(1+roc.iloc[i]))
		else:
			nv.append(nv[i-1])
	NVI=pd.Series(nv,name='NVI')
	return NVI

def pvi(close,volume):
	"""
	  Positive volume index (PVI)
	Inputs: 
	        Input    | Type                             | Description
	       =========================================================================================
	         close   |pandas.DataFrame                  | Prices
	         volume  |pandas.DataFrame                  | Volumes
	Outputs:
	        Output | Type                             | Description
	       ========================================================================================
	               | pandas.Series                    | PVI 
	"""
	roc=pd.Series(close.diff(1)/close.shift(1), name="ROC")
	pv=[np.nan,roc.iloc[1]]
	for i in range(2,len(volume)):
		if volume[i] > volume[i-1]:
			pv.append(pv[i-1]*(1+roc.iloc[i]))
		else:
			pv.append(pv[i-1])
	PVI=pd.Series(pv,name='PVI')
	return PVI
