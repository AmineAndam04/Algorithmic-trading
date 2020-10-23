import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from .Indicators import *

 
def plot_mms(df,n):
	"""
	Graphe de Moyenne Mobile simple 

	"""
	fig = plt.figure()
	ax1 = fig.add_subplot(111, ylabel='MASI')
	MMS=mms(df,n)
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	MMS['MMS'].plot(ax=ax1, color='r', lw=2., legend=True)
	plt.legend(loc=0)
	plt.show()

def plot_mme(df,n):
    """
    Graphe moyenne mobile exponentielle 
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='MASI')
    MME=mme(df,n)
    df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
    MME['MME'].plot(ax=ax1, color='r', lw=2., legend=True)
    plt.legend(loc=0)
    plt.show()

def plot_macd(df,ws,wl, wsig=9):
	"""

	"""
	MACD=macd(df,ws,wl,wsig)
	MACD['hist']=MACD["MACD"]-MACD['MACDsignal']
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(312, ylabel='MACD')
	MACD['MACD'].plot(ax=ax2, color='black', lw=2., legend=True)
	MACD["MACDsignal"].plot(ax=ax2, color='g', lw=2., legend=True)
	ax3 = fig.add_subplot(313, ylabel='MACD')
	MACD["hist"].plot(ax=ax3, color='r', linestyle="--",grid=True, legend=True,use_index=False)
	plt.show()

def plot_rsi(df,n):
    """

    """
    rs=rsi(df,n)["RSI"]*100
    fig = plt.figure()
    ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
    df.plot(ax=ax1, color='black', lw=2., legend=True,figsize=(13,9))
    ax2=fig.add_subplot(312, ylabel='RSI')
    rs.plot(ax=ax2, color='b', lw=2., legend=True)
    plt.show()

def plot_bollinger(df,w,k):
	"""

	"""
	BB=bollinger(df,w,k)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	BB[["COURS_CLOTURE","BBUP","BBDOWN"]].plot(ax=ax1,figsize=(13,9))
	ax1.fill_between(df.index,BB["BBUP"],BB["BBDOWN"],facecolor='red', alpha=0.4)


def plot_momentum(df,w,wsig):
	"""

	"""
	MOM=momentum(df,w,wsig)[['MOM','MOMsignal']]
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='Momentum ')
	MOM["MOM"].plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	MOM["MOMsignal"].plot(ax=ax2, color='r', lw=2., legend=True)
	plt.show()

def plot_roc(df,w):
	"""

	"""
	roc=rate_of_change(df,w)['ROC']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='ROC')
	roc.plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	plt.show()

def plot_stochastique(df ,high,low, n , w):
	"""

	"""
	stoch=stochastique(df ,high,low, n , w)[['%K','%D']]
	fig = plt.figure()
	ax1 = fig.add_subplot(311, ylabel='COURS_CLOTURE')
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

	"""
	ob=obv(df,vol)['OBV']
	obvsignal=pd.Series(ob.rolling(10).mean(),name= 'obvsignal')
	ob.plot(legend=True, figsize=(10,5))
	obvsignal.plot(legend=True)

def plot_williams(df,n):
	"""

	"""
	wil=williams(df,n)['%R']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	df.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='Williams %R')
	wil.plot(ax=ax2, color='b', lw=2., legend=True)
	plt.axhline(-20,color="green")
	plt.axhline(-80,color="red")
	plt.show()

def plot_MFI(close,volume,high,low,n):
	"""

	"""
	mfi=MFI(close,volume,high,low,n)['MFI']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='MFI')
	mfi.plot(ax=ax2, color='b', lw=2., legend=True)
	plt.show()

def plot_cho(close,volume,high,low,n,ws,wl):
	"""

	"""
	ch=cho(close,volume,high,low,n,ws,wl)['CHO']
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='CHO')
	ch.plot(ax=ax2, color='b', lw=2., legend=True,grid=True)
	plt.show()


def plot_nvi(close,volume):
	"""

	"""
	nv=nvi(close,volume)
	nvisignal=pd.Series(nv.rolling(10).mean(), name= "NVIsignal")
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='NVI')
	nv.plot(ax=ax2, lw=2., legend=True,grid=True)
	nvisignal.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.show()

def plot_pvi(close,volume):
	"""

	"""
	pv=pvi(close,volume)
	pvisignal=pd.Series(pv.rolling(10).mean(), name= "PVIsignal")
	fig = plt.figure()
	ax1 = fig.add_subplot(211, ylabel='COURS_CLOTURE')
	close.plot(ax=ax1, color='g', lw=2., legend=True,figsize=(13,9))
	ax2 = fig.add_subplot(212, ylabel='PVI')
	pv.plot(ax=ax2, lw=2., legend=True,grid=True)
	pvisignal.plot(ax=ax2, lw=2., legend=True,grid=True)
	plt.show()
