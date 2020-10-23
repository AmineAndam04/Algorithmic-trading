import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *

 

def plt_mms1(df,n):
    """
     R1

    """
    signal=pd.DataFrame(index=df.index)
    signal['signal']=0
    signal["compa"]=0
    MMS=mms(df,n)["MMS"]
    signal["compa"].loc[n:]=np.where(df[n:] > MMS[n:],1,0)
    signal["signal"]=adjustsignal(signal["compa"].diff())
    pmval=pmv(df,signal["signal"])
    pmval=pd.Series(pmval,index=df.index)
    fig = plt.figure()
    ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
    df.plot(ax=ax1, color='g', lw=.5)
    MMS.plot(ax=ax1, lw=.5, figsize=(13,9))
    ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
    ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
    plt.legend(["COURS_CLOTURE","MMS","Achat","Vente"])  
    ax2 = fig.add_subplot(212, ylabel='PMV')
    ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
    ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
    plt.legend(["Plus_value","Moins_value"])

def plt_mms2(df,ws,wl):
	"""
	R2

	"""
	signal=pd.DataFrame(index=df.index)
	signal['compa']=np.nan
	signal["MMS_COURT"]=mms(df,ws)["MMS"]
	signal["MMS_LONG"]=mms(df,wl)["MMS"]
	signal["compa"].loc[wl:]=np.where(signal["MMS_COURT"][wl:] > signal["MMS_LONG"][wl:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	signal["MMS_COURT"].plot(ax=ax1, lw=2.)
	signal["MMS_LONG"].plot(ax=ax1, lw=2.)
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","MMS_COURT","MMS_LONG","Achat","Vente"])
	plt.title("Double Moving Average Trading Strategy")
	ax2 = fig.add_subplot(212, ylabel='PMV')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_macd1(df,ws,wl):
	"""


	"""
	signal=pd.DataFrame(index=df.index)
	MACD=macd(df,ws,wl)["MACD"]
	signal["compa"]=np.nan
	signal["compa"][wl:]=np.where(MACD[wl:]>0 ,1 ,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.title("MACD Strategy")
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD.plot(ax=ax2, color='black', lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])



def plt_macd2(df,ws,wl,wsig=9):
	"""

	"""
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	MACD=macd(df,ws,wl,wsig)[["MACD","MACDsignal"]]
	signal["compa"][wl:]=np.where(MACD["MACD"][wl:] > MACD["MACDsignal"][wl:] ,1 ,0 )
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.title("MACD Strategy")
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def  plt_rsi(df,n):
	"""

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
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal ==-1], 'v', markersize=7, color='r')
	plt.title("RSI Strategy")
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='RSI')
	rs.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(0.7,color="green")
	plt.axhline(0.3,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_bollinger(df,w,k):
	"""

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
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.fill_between(df.index,bb["BBUP"],bb["BBDOWN"],facecolor='red', alpha=0.2)
	bb.plot(ax=ax1, lw=.5)
	ax1.plot(signal.loc[signal.sig==1].index ,df[signal.sig==1],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.sig==-1].index ,df[signal.sig==-1],'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","BBDOWN","BBUP","Achat",  "Vente"])
	plt.title("Bondes de Bollinger Trading Strategy")
	ax2=fig.add_subplot(212, ylabel='PMV')
	ax2.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax2.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_momentum(df,w,wsig=9):
	"""

	"""
	MOM=momentum(df,w,wsig=9)[["MOM","MOMsignal"]]
	signal=pd.DataFrame(index=df.index)
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where(MOM["MOM"][w:] > MOM["MOMsignal"][w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2=fig.add_subplot(312, ylabel='MOM')
	MOM.plot(ax=ax2, legend=True, grid=True)
	plt.title("MOM Strategy")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_stochastique1(df,high,low ,n , w):
	"""

	"""
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
	signal["signal"]=sig
	pmval=pmv(df,sig)
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Stochastique')
	k.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_stochastique2(df,high,low,n , w):
	"""

	"""
	signal=pd.DataFrame(index=df.index)
	k=stochastique(df ,high,low, n , w)[["%K","%D"]]
	signal["compa"]=np.nan
	signal["compa"][n+w :]=np.where(k["%K"][n+w:] > k["%D"][n+w:],1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Stochastique')
	k.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_obv(df,vol,n):
	"""

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
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='ON-Banlance Volume')
	ob.plot(ax=ax2, lw=2., legend=True,grid=True)
	obs.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_roc(df,w):
	"""

	"""
	signal=pd.DataFrame(index=df.index)
	roc=rate_of_change(df,w)["ROC"]
	signal["compa"]=np.nan
	signal["compa"][w:]=np.where( roc[w:]> 0,1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(df,signal["signal"])
	pmval=pd.Series(pmval,index=df.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,df[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,df[signal.signal == -1 ], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Rate of change')
	roc.plot(ax=ax2, lw=2., legend=True,grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(df.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(df.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_mfi(close,volume,high,low,n):
	"""

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
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='k', lw=.5, figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal== 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='MFI')
	mfi.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.axhline(80,color="green")
	plt.axhline(20,color="red")
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_cho(close,volume,high,low,n,ws,wl):
	"""

	"""
	signal=pd.DataFrame(index=close.index)
	signal["compa"]=np.nan 
	ch=cho(close,volume,high,low,n,ws,wl)["CHO"]
	signal["compa"][ws+wl:]=np.where(ch[ws+wl:] > 0, 1,0)
	signal["signal"]=adjustsignal(signal["compa"].diff())
	pmval=pmv(close,signal["signal"])
	pmval=pd.Series(pmval,index=close.index)
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","MMS","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Chaikin Oscillator')
	ch.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])


def plt_nvi(close,volume,n):
	"""

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
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","MMS","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Negative Volume Index')
	nv.plot(ax=ax2,lw=2., legend=True, grid=True)
	nvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

def plt_pvi(close,volume,n):
	"""

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
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=.5,figsize=(13,9))
	ax1.plot(signal.loc[signal.signal== 1.0].index ,close[signal.signal == 1.0],'^', markersize=7, color='g')
	ax1.plot(signal.loc[signal.signal== -1].index,close[signal.signal == -1.0], 'v', markersize=7, color='r')
	plt.legend(["COURS_CLOTURE","MMS","Achat","Vente"])
	ax2 = fig.add_subplot(312, ylabel='Negative Volume Index')
	pv.plot(ax=ax2,lw=2., legend=True, grid=True)
	pvis.plot(ax=ax2,lw=2., legend=True, grid=True)
	ax3=fig.add_subplot(313, ylabel='PMV')
	ax3.fill_between(close.index,pmval,where=(pmval > 0), facecolor='green', alpha=0.5)
	ax3.fill_between(close.index,pmval,where=(pmval < 0), facecolor='red',alpha=0.5)
	plt.legend(["Plus_value","Moins_value"])

