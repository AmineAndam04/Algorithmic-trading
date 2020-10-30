import numpy as np




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


def dietz_court(signal,close):
    date=close.index
    date=date.values
    close=np.array(close)
    signal=np.array(signal.fillna(0))
    qti=signal.cumsum()
    CF=signal*close
    V1=qti[-1]*close[-1]
    V0=CF[0]
    w=[]
    for i in range(len(close)):
        w.append((date[-1]-date[i])/(date[-1]-date[0]))
    dcourt=(V1-V0-CF[1:].sum())/(V0+(w[1:]*CF[1:]).sum())
    return dcourt