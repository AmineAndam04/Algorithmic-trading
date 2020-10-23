import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def mms(df,n):
	"""
	Moyenne Mobile Simple  
	"""
	MMS = pd.Series(df.rolling(n, min_periods=n).mean(), name='MMS')
	df=pd.DataFrame(df)
	df=df.join(MMS)
	return df


def mme(df,n):
	"""
	Moyenne Mobile exponentielle
	 :Paramètre:
	   df(pandas.DataFrame/ pandas.Series): la séries des prix ou autre 
	   n (int): période


	"""
	exp=[df[:n].mean()]
	lamda=2/(1+n)
	for i in range(1,len(df)-n+1):
		val=(1-lamda)*exp[i-1]+lamda*df[i+n-1]
		exp.append(val)
	MME=pd.Series(index=df.index)
	MME[n-1:]=exp 
	MME.name="MME"
	return MME

	
def macd(df,ws,wl, wsig=9):
	"""
	Moving Average Convergence Divegence
	  :Paramère:
	    df: 
	    ws: ordre de court terme
	    wl: ordre de long terme
	    wsig: ordre pour le signal line
	  :return:
	   pndas.DataFrame  contient les valeurs des MACD et le Signal line

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
 	 Relative Strength index
 	  :Paramètre:
 	   df: pandas.DataFrame
 	   n : ordre
 	  :return:
 	   pandas.DataFrame

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
 	df.columns=["COURS_CLOTURE","RSI"] 
 	return df
def bollinger(df,w,k):
 	"""
 	Bandes de Bollinger
 	 Paramètre: df: pandas.DataFrame ou pandas.Series: vecteur des prix
 	            w : ordre de la moyenne mobile 
 	            k : 
 	 Retour:  BBDOWN: bande inférieure
 	          BBUP  : bande supérieure
 	          BBMID : bande au milieu


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
 	df.columns=['COURS_CLOTURE',"BBDOWN","BBMID","BBUP"]
 	return df
def momentum(df,w,wsig=9):
	"""
	Momentum
	  Paramètre: df: le  vecteur des prix
	              w: ordre
	              wsig ; ordre de signal ligne
	  Retour:   Momentume (pndas.DataFrame)


	"""
	MOM=pd.Series(df.diff(w),name="MOM")
	MOMsignal=pd.Series(MOM.rolling(wsig, min_periods=wsig).mean(), name= "MOMsignal")
	df=pd.DataFrame(df)
	df=df.join(MOM)
	df=df.join(MOMsignal)
	return df

def rate_of_change(df,w):
	"""
	 :Paramètre:
	   df: DataFrame
	   w : int 

	"""
	ROC=pd.Series(100*(df.diff(w))/df.shift(w), name='ROC')
	df=pd.DataFrame(df)
	df=df.join(ROC)
	return df

def stochastique(df ,high,low, n , w):
	"""
	Paramètre :  df : pndas.DataFrame des prix
	             high : 
	             low: 
	             n(int) : ordre de %K
	             w(int) : ordre de %D

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
	On Balance Volume
	 :Paramètre: 
	  df : pandas.DataFrame: les prix
	  vol: pandas.DataFrame : les volumes

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
	   :Paramètre: df : pandas.DataFrame
	               n  : ordre 

	"""
	PH=df.rolling(n, min_periods=n).max()
	PB=df.rolling(n, min_periods=n).min()
	R=pd.Series(-100*(PH-df)/(PH-PB), name='%R')
	df=pd.DataFrame(df)
	df=df.join(R)
	return df

def MFI(close,volume,high,low,n):
	"""
	  Money Flow Index
	   :Paramètre: close: prix de cloture
	               volume: les volume
	               high: prix les plus haut durant la séance
	               low: prix les plus bas durant la séance
	               n : ordre

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
	  :Paramètre: close: prix de cloture
	               volume: les volume
	               high: prix les plus haut durant la séance
	               low: prix les plus bas durant la séance
	               n : ordre
	               ws :ordre court terme
	               wl : ordre long terme 

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
	Negative Volume index
	  :Paramètre: close: prix de cloture
	              volume:  les volumes 

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
	  Positive volume index
	    :Paramètre:  close : prix de cloture
	                 volume : les volume


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
