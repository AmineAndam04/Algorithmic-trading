import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *


def signal_mms1(df,n):
	signal=pd.DataFrame(index=df.index)
	signal['signal']=0
	signal["compa"]=0
	MMS=mms(df,n)["MMS"]
	signal["compa"].loc[n:]=np.where(df[n:] > MMS[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_mms2(df,ws,wl):
	signal=pd.DataFrame(index=df.index)
	signal['compa']=np.nan
	signal["MMS_COURT"]=mms(df,ws)["MMS"]
	signal["MMS_LONG"]=mms(df,wl)["MMS"]
	signal["compa"].loc[wl:]=np.where(signal["MMS_COURT"][wl:] > signal["MMS_LONG"][wl:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_macd1(df,ws,wl):
	signal=pd.DataFrame(index=df.index)
	MACD=macd(df,ws,wl)["MACD"]
	signal["compa"]=np.nan
	signal["compa"][wl:]=np.where(MACD[wl:]>0 ,1 ,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_macd2(df,ws,wl,wsig=9):
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	MACD=macd(df,ws,wl,wsig)[["MACD","MACDsignal"]]
	signal["compa"][wl:]=np.where(MACD["MACD"][wl:] > MACD["MACDsignal"][wl:] ,1 ,0 )
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def  signal_rsi(df,n):
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
	MOM=momentum(df,w,wsig=9)[["MOM","MOMsignal"]]
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where(MOM["MOM"][w:] > MOM["MOMsignal"][w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_stochastique1(df,high,low ,n , w):
	signal=pd.DataFrame(index=df.index)
	k=stochastique(df ,high,low, n , w)["%K"]
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
def signal_stochastique2(df,high,low ,n , w):
	signal=pd.DataFrame(index=df.index)
	k=stochastique(df ,high,low, n , w)[["%K","%D"]]
	signal["compa"]=np.nan
	signal["compa"][n+w :]=np.where(k["%K"][n+w:] > k["%D"][n+w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_obv(df,vol,n):
	signal=pd.DataFrame(index=df.index)
	ob=obv(df,vol)["OBV"]
	obs=pd.Series(ob.rolling(n).mean(), name='obvsignal')
	signal["compa"]=np.nan 
	signal["compa"][n:]=np.where( ob[n:]  > obs[n:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_roc(df,w):
	signal=pd.DataFrame(index=df.index)
	roc=rate_of_change(df,w)["ROC"]
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where( roc[w:]> 0,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return  signal["signal"]
def signal_mfi(df,vol,high,low,n):
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
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	ch=cho(df,vol,high,low,n,ws,wl)["CHO"]
	signal["compa"][ws+wl:]=np.where(ch[ws+wl:] > 0, 1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_nvi(df,vol,n):
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	nv=nvi(df,vol)
	nvis=pd.Series(nv.rolling(n).mean(), name="NVIsignal")
	signal["compa"][n:]=np.where(nv[n:] > nvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]
def signal_pvi(df,vol,n):
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan 
	pv=pvi(df,vol)
	pvis=pd.Series(pv.rolling(n).mean(), name="PVIsignal")
	signal["compa"][n:]=np.where(pv[n:] > pvis[n:] ,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	return signal["signal"]

## Function to adjust signals: if we have a sell signal but we don't have something to sell, we ignore that signal
def adjustsignal(signal):
	sig=[]
	qtite=0
	for i in signal:
		if i > 0 :
			sig.append(i)
			qtite+=1
		elif i < 0:
			if qtite >= abs(i) :
				sig.append(i)
				qtite+=-i
			else:
				sig.append(0)
		else:
			sig.append(0)
	return sig