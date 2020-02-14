# util.py

'''
PCI algorithm borrowed from # Leonardo Barbosa ()leonardo.barbosa@usp.br) and adapted.
'''

import numpy as np
import pylab as plt 
import time 
#import bitarray
from bitarray import bitarray

from progress.bar import Bar

import mne 
from mne.minimum_norm import apply_inverse, apply_inverse_epochs, make_inverse_operator  

params=dict(Nboot=500,tblMIN=-400,tblMAX=-5,alpha=0.01,snr=5.,method='dSPM',perc=0.01)  

BaseSort=2.

SpeedUp=False

def pci_norm_factor(D):
    '''
    '''
    L = D.shape[0] * D.shape[1]
    p1 = sum(1.0 * (D.flatten() == 1)) / L
    p0 = 1 - p1
    if p1*p0:
        H = -p1 * np.log2(p1) -p0 * np.log2(p0)
    else:
        print('p0=%g\np1=%g\n'%(p0,p1))
        H=0.
    S = (L * H) / np.log2(L)
    return S

def lz_complexity_2D(D):
    '''
    '''
    #global ct   # time dependent complexity 
    if SpeedUp:  
        ''' sort activity and remove zero lines 
        '''
        # last check 
        if D.max()==0: 
            ct = np.repeat(0,D.shape[1])
            return 0

        Irank=np.sum(D,axis=1).argsort()
        S=D[Irank,:].sum(axis=1)
        Izero=np.where(S==0)[0]
        if len(Izero)>1:
            Dnew=D[Irank,:][Izero[-1]+1:,:] # +1 as a parameter ?
        else:
            Dnew=np.copy(D[Irank,:])
    else:
        Dnew=np.copy(D)

    if len(Dnew.shape) != 2:
        raise Exception('data has to be 2D!')

    # initialize
    (L1, L2) = Dnew.shape

    c=1; r=1; q=1; k=1; i=1
    stop = False

    # convert each column to a sequence of bits
    bits = [None] * L2
    for y in range(0,L2):
        bits[y] = bitarray(Dnew[:,y].tolist())

    # action to perform every time it reaches the end of the column
    def end_of_column(r, c, i, q, k, stop):
        r += 1
        if r > L2:
            c += 1
            stop = True
        else:
            i = 0
            q = r - 1
            k = 1
        return r, c, i, q, k, stop

    ct=[]

    # main loop
    while not stop:

        if q == r:
            a = i+k-1
        else:
            a=L1
        found = not not bits[q-1][0:a].search(bits[r-1][i:i+k], 1)
        if found:
            k += 1
            if i+k > L1:
                (r, c, i, q, k, stop) = end_of_column(r, c, i, q, k, stop)
                ct.append(c)     
        else:
            q -= 1
            if q < 1:
                c += 1
                i = i + k
                if i + 1 > L1:
                    (r, c, i, q, k, stop) = end_of_column(r, c, i, q, k, stop)
                    ct.append(c)      
                else:
                    q = r
                    k = 1
    #return c
    return ct

def PCIlz(D):
    return lz_complexity_2D(D) / pci_norm_factor(D)

def sort_binJ(binJ,whatsort='STD'):
    '''
    '''
    _,nTIME=binJ.shape

    if whatsort=='STD':
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsort=binJ[Irank,:]

    if (whatsort=='EG1') or (whatsort=='EG'): # the standard EG=EG1 
        binCODE=BaseSort**np.arange(nTIME) 
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsortTMP=binJ[Irank,:]  
        # sort the blocks 
        SumCh=np.sum(binJsortTMP,axis=1)
        Su=np.unique(SumCh)    
        binJ2c=np.copy(binJsortTMP)
        for S in Su:
            I=np.where(SumCh==S)[0]
            y=np.sum(binJsortTMP[I,:]*binCODE,axis=1)
            J=np.argsort(y)[::-1]
            binJsortTMP[I,:]=binJ2c[I[J],:]
        binJsort=binJsortTMP

    if whatsort=='EG2': # in analogy to S2 
        binCODE=BaseSort**(nTIME-np.arange(nTIME))
        SumCh=np.sum(binJ,axis=1)
        Irank=SumCh.argsort()[::-1]
        binJsortTMP=binJ[Irank,:]
        # sort the blocks
        SumCh=np.sum(binJsortTMP,axis=1)
        Su=np.unique(SumCh)    
        binJ2c=np.copy(binJsortTMP)
        for S in Su:
            I=np.where(SumCh==S)[0]
            y=np.sum(binJsortTMP[I,:]*binCODE,axis=1)
            J=np.argsort(y)[::-1]
            binJsortTMP[I,:]=binJ2c[I[J],:]
        binJsort=binJsortTMP

    if whatsort=='S1':
        binCODE=BaseSort**np.arange(nTIME) 
        y=np.sum(binJ*binCODE,axis=1)  
        J=np.argsort(y)[::-1]
        binJsort=binJ[J,:]
                    
    if whatsort=='S2':
        binCODE=BaseSort**(nTIME-np.arange(nTIME))
        y=np.sum(binJ*binCODE,axis=1)  
        J=np.argsort(y)[::-1]
        binJsort=binJ[J,:]

    return binJsort


def compute_inverse(fnd):
    ''' move it elsewhere ... 
    '''
    # load EEG
    dataEEG = mne.read_epochs(fnd)
    dataEEG.apply_baseline(baseline=(None,0))  # baseline correction (important to have accurate estimates of the covariance matrix)
    info = mne.io.read_info(fnd)
    # load forward model 
    fwd = mne.read_forward_solution(fn_forward)
    lambda2 = 1.0 / snr ** 2
    # prepare inverse 
    cov = mne.make_ad_hoc_cov(info)
    inverse_operator = make_inverse_operator(info, fwd, cov, depth=None, fixed=True )  
    # 
    if decimate>0: dataEEG.decimate(decimate)
    evoked = dataEEG.average()
    stcEVK = apply_inverse(evoked, inverse_operator, lambda2, method = method, verbose = False )
    return dataEEG,stcEVK,info,inverse_operator


def iAAFT(X,max_it=500,prev_err=1e10):
    """
        Generate surrogate data with matching amplitude spectrum and
        signal distribution. Based on iAAFT_TD (Lucio, Valdes and Rodriguez,
        "Improvements to surrogate data methods for nonstationary time series",
        PRE 85 (2012).)
        X = time-series
    """
    pp=X.size 
    Xsorted=np.sort(X)
    gt=X[0]+(X[-1]-X[0])*np.linspace(0,pp-1,pp)/(pp-1) 
    Xdt=X-gt #detrend
    Sdt=np.fft.rfft(Xdt) 
    Sdtangle=np.angle(Sdt)  # original angles
    Sdtamp=np.abs(Sdt)      # original amplitudes
    S2dtangle=np.random.permutation(Sdtangle) # initial condition: permuted phases
    X2=np.real(np.fft.irfft(Sdtamp * np.exp(1j*S2dtangle), n=pp))+gt    
    #start iterations
    E=np.zeros(max_it)
    c=0
    err=prev_err-1
    while (prev_err>err) and (c<max_it):
        X2dt=X2-gt #detrend
        S2dt=np.fft.rfft(X2dt,n=pp) 
        S2dtangle=np.angle(S2dt)
        X3=np.real(np.fft.irfft(Sdtamp * np.exp(1j*S2dtangle), n=pp))+gt 
        INDs=np.argsort(X3)
        X2[INDs]=Xsorted
        prev_err=err
        A2=np.abs(S2dt)
        err=np.mean(abs(A2-Sdtamp))
        E[c]=err
        c=c+1
    return X3

def trim_matrix(A,sortYN=True):
    '''
    '''
    I=np.argmax(np.abs(A),axis=0)  # max in time of each channel, it is Ado's [~,I]= ... 
    if sortYN: I=np.sort(I)
    J=np.where(np.diff(I)==0)[0]
    I=np.delete(I,J)
    return A[I,:],I

def Build_binJ(t,J,Threshold,plotIt=False):
    '''
        J (nSRC x nTIME) 
    '''
    if not(J.shape==Threshold.shape):
        # assume Threshold is of size (nSRC,1)
        ncol=J.shape[1]
        Threshold=np.kron(np.ones((1,ncol)),Threshold) 
    # threshold J        
    binJ=np.array(np.abs(J)>Threshold,dtype=int)
    if plotIt:
        I,K=np.where(binJ)
        plt.figure()
        plt.plot(t[K],I,'ko')
    return binJ

def Threshold_IAFFT(J,t,params): 
    ''' tblMAX, where baseline finishes 
    '''
    global lun0_IAAFT
    nSRC,nTIME=J.shape # evoked (averaged across trials)
    perc=params['perc']
    Nboot=params['Nboot']
    tblMIN=params['tblMIN']
    tblMAX=params['tblMAX']
    alpha=params['alpha']

    Ibl=np.where((t>tblMIN)&(t<tblMAX))[0]
    N0=len(Ibl)
    Jbl=J[:,Ibl]    # baseline currents  

    basecorr = np.mean(Jbl,axis=1)           # mean baseline of prestimulus activity       
    Norm = np.std(Jbl,axis=1,ddof=1)         # standard deviation of baseline of prestimulus activity       

    ind0=np.where(Norm==0)[0]
    if len(ind0):
        lun0=len(ind0)
        lun0_IAAFT=lun0
        print('%d channels out of %d have zero Norm !!! \n'%(lun0,nSRC))
        print('set the zero Norm values equal to 1e-12 !!! \n')
        Norm[ind0]=1e-12
    else:
        lun0_IAAFT=0

    NUM = np.kron(np.ones((1,N0)),basecorr.reshape((nSRC,1)))
    DEN = np.kron(np.ones((1,N0)),Norm.reshape((nSRC,1)))

    normJ=(Jbl-NUM)/DEN                     # normalized J, nJ in Ado's code  
    if perc<1:
        num_min=int(nSRC*perc)
        normJ3=np.copy(normJ)
        normJ2,I=trim_matrix(normJ)
        k=0
        while normJ2.shape[0]<num_min:
            normJ3=np.delete(normJ3,I,axis=0)    
            _,J=trim_matrix(normJ3)
            normJ2=np.vstack((normJ2,normJ[J,:]))
            k+=1 
    else:
        print('Use all sources for bootstrap (can be slow!)')
        normJ2=np.copy(normJ)
    T=np.zeros((Nboot,N0))
    X=np.zeros(normJ2.shape)
    bar=Bar('Processing',max=Nboot)
    for perm in range(Nboot):
        bar.next()
        for i in range(normJ2.shape[0]): 
            X[i,:]=iAAFT(normJ2[i,:])
        T[perm,:]=np.max(np.abs(X),axis=0)  
    bar.finish()    
    T=np.sort(T.reshape(Nboot*N0))  
    calpha=1-alpha
    calpha_index=int(np.round(calpha*Nboot*N0))-1 
    TT=Norm*T[calpha_index]
    # free some space
    norm,normJ2,normJ3=[],[],[]    
    X,T=[],[]    
    return TT.reshape((nSRC,1)) # it might be more useful to return 

def Threshold_Baseline(J,t,dataEEG,info,inv,params):
    '''
        stcEVK      nSRC x nTIME            -> J 
        dataEEG     nTR x nCH x nTIME      
        info        
        inv 
    '''
    global lun0_Baseline

    Nboot=params['Nboot']
    tblMIN=params['tblMIN']
    tblMAX=params['tblMAX']
    alpha=params['alpha']
    snr=params['snr']
    method=params['method']

    Y = dataEEG.get_data()
    nTR,nCH,_ = Y.shape 
    nSRC,nTIME = J.shape 

    Ibl=np.where((t>tblMIN)&(t<tblMAX))[0]  # baseline 
    N0=len(Ibl)

    lambda2 = 1.0 / snr ** 2
    baseline=J[:,Ibl]                       
    basecorr=np.mean(baseline,axis=1)       # mean of baseline for each channel   
    Norm=np.std(baseline,axis=1,ddof=1)     # std of baseline for each channel           
    # 
    ind0=np.where(Norm==0)[0]
    if len(ind0):
        lun0=len(ind0)
        lun0_Baseline=lun0
        print('%d channels out of %d have zero Norm !!! \n'%(lun0,nSRC))
        print('set the zero Norm values equal to 1e-12 !!! \n')
        Norm[ind0]=1e-12
    else:
        lun0_Baseline=0
    #
    NUM = np.kron(np.ones((1,N0)),basecorr.reshape((nSRC,1)))
    DEN = np.kron(np.ones((1,N0)),Norm.reshape((nSRC,1)))
    # 
    randontrialsT=np.random.randint(0,nTR,nTR)
    Bootstraps=np.zeros((Nboot,N0))          
    bar=Bar('Processing',max=Nboot)
    for per in range(Nboot):
        bar.next()
        YT=np.zeros((nTR,nCH,N0))
        for j in range(nTR):
            randonsampT = np.random.choice(Ibl,N0,replace=True) #np.random.randint(0,N0,N0) 
            YT[j] = Y[randontrialsT[j]][:,randonsampT]
        YTE = mne.EpochsArray(YT,info,verbose=False)
        ET=apply_inverse(YTE.average() ,inv ,method=method ,lambda2=lambda2 ,verbose=False ).data   # pick_ori=pick_ori,
        ET=(ET-NUM)/DEN     # computes a Z-value
        Bootstraps[per,:] = np.max(np.abs(ET),axis=0) # maximum statistics in space 
    bar.finish() 
    # computes threshold
    Bootstraps=np.sort(np.reshape(Bootstraps,(Nboot*N0)))
    soglia2=1-alpha 
    soglia2=int(np.round(soglia2*Nboot*N0))-1                   
    TT=Norm*Bootstraps[soglia2]                                 # threshold !
    # free some space
    norm,normJ2,normJ3=[],[],[]    
    ET,YTE,YT=[],[],[]
    Bootstraps=[]
    return TT.reshape((nSRC,1))




