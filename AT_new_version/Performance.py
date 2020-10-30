import pandas as pd 
import numpy as np 
from .utils import dietz_court


def pmv(close,signal):
	"""
		La Plus-ou-moins-value

	"""
	close=np.array(close)
	signal=np.array(signal.fillna(0))
	qt=signal.cumsum()
	achat= np.where(signal>=1,1,0)
	PDR=[]
	for t in range(len(close)):
		p=(close[:t+1]*achat[:t+1]).sum()
		if achat[:t+1].sum()==0 :
			val=0
		else :
			p=p/(achat[:t+1].sum())
			val=qt[t]*p 
		PDR.append(val)
	PMV_lat=close*qt-PDR
	PMV_re=[0]
	for i in range(1,len(close)):
		vl=PDR[i]-PDR[i-1]+PMV_re[i-1]-close[i]*signal[i]
		PMV_re.append(vl)
	PMV=PMV_lat+PMV_re
	return PMV



def Dietz(close,signal):
    """
    Modified Dietz Return
    """
    qt=signal.cumsum().round(3)
    diet=[]
    i=0
    while i < len(qt):
        debut=i
        while i< len(qt) and qt[i]!=0  :
            i+=1
        if debut !=i:
            fin=i+1
            val=dietz_court(signal.iloc[debut:fin],close[debut:fin])
            diet.append(val)
        i+=1
    diet=np.array(diet)+1
    return (diet.prod()-1)


