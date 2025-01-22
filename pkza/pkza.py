import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from pysbt import sbt

class pkza:
    '''pkza'''
    # Calculate Zel'dovich power spectrum (Mohammed & Seljak 2014):
    #   p(k) = 4pi int dq q^2  int du cos(kqu) [exp{-1/2 k^2X(q)} exp{-1/2 k^2 u^2 Y(q)} - exp{-1/2 k^2 X0}],
    #        = 4pi int dq q^2 { [exp(-1/2 k^2(X+Y))-exp(-1/2 k^2 X0)] j0(kq) 
    #          + exp(-1/2 k^2(X+Y)) Sum_n>=1 [k^2Y/(kq)]^n jn(kq)}
    #
    # where
    #
    #   X(q) = 1/(2pi^2) int dk Plin(k) [ 2/3 - 2 * j1(kq) / (kq) ]
    #
    #   Y(q) = 1/(2pi^2) int dk Plin(k) [ -2 * j0(kq) + 6 * j1(kq) / (kq) ]
    #
    #     X0 = lim q--> X(q)
    #
    # and Plin(k) is the linear power spectrum
    def __init__(self, **kwargs):
        def _kfilterd(k,k0):
            val = np.exp(-k**2/k0**2)
            if k > 5*k0: val=0
            return val
        _kfilterd = np.vectorize(_kfilterd)

        import pkg_resources
        pkfiled = pkg_resources.resource_filename('pkza', 'data/matterpower.dat')
        self.plinfile = kwargs.get('plinfile', pkfiled)
        self.nintk    = kwargs.get('nintk',    3000)
        self.kfilter  = kwargs.get('kfilter',  _kfilterd)
        self._get_plin()

    # get input linear pspec
    def _get_plin(self):
        ki, plini = np.loadtxt(self.plinfile, unpack=True)
        phigh = sp.interpolate.interp1d(np.log(ki),plini,bounds_error=False)
        plow = lambda k: np.exp(np.log(plini[0])+(np.log(k)-np.log(ki[0]))*(np.log(plini[1])-np.log(plini[0]))/(np.log(ki[1])-np.log(ki[0])))
        self.ka0 = np.logspace(-7,np.log10(ki[-1]),self.nintk)
        self.plin0 = phigh(np.log(self.ka0))
        self.plin0[self.ka0<ki[0]] = plow(self.ka0[self.ka0<ki[0]])

    def _truncatepspec(self,k0):
        plina = self.plin0 + 0.0
        plina *= self.kfilter(self.ka0,k0)
        return sp.interpolate.interp1d(np.log(self.ka0),plina)

    def _getXYZ(self,k0):
        # MS14 X0, X(q) and Y(q) using sbt code / sp quad

        plin = self._truncatepspec(k0)
        ka = self.ka0[plin(np.log(self.ka0))>0]
        plina = plin(np.log(ka))
        ss = sbt(ka,kmax=1e5)
        qa = ss.kk
        lqa = np.log(qa)

        dj0 = plina / ka**2
        dj1 = plina / ka**3
        dpi = lambda lnk: plin(lnk)*np.exp(lnk)

        j0 = 1/(2*np.pi**2) * ss.run(dj0, direction=1, norm=False, l=0)
        j1 = 1/(2*np.pi**2) * ss.run(dj1, direction=1, norm=False, l=1) / qa
        pi = 1/(2*np.pi**2) * sp.integrate.quad(dpi, np.log(ka[0]), np.log(ka[-1]),limit=200,full_output=1)[0]

        Xs = 2 / 3 * pi - 2 * j1
        Ys = -2 * j0 + 6 * j1

        X0 = Xs[-1]
        X = sp.interpolate.interp1d(np.log(qa),Xs)
        Y = sp.interpolate.interp1d(np.log(qa),Ys)
        return qa,lqa,X0,X,Y,plin

    def pzel(self,kv,k0,N):
        #  Zel'dovich power spectrum is given by (MS14)
        #   p(k) = 4pi int dq q^2  int du cos(kqu) [exp{-1/2 k^2X(q)} exp{-1/2 k^2 u^2 Y(q)} - exp{-1/2 k^2 X0}],
        #        = 4pi int dq q^2 { [exp(-1/2 k^2(X+Y))-exp(-k^2 X0)] j0(kq) 
        #          + exp(-1/2 k^2(X+Y)) Sum_n>=1 [k^2Y/(kq)]^n jn(kq)}
        #
        #   X(q) = 1/(2pi^2) int dk Plin(k) [ 2/3 - 2 * j1(kq) / (kq) ]
        #
        #   Y(q) = 1/(2pi^2) int dk Plin(k) [ -2 * j0(kq) + 6 * j1(kq) / (kq) ]
        #
        #     X0 = lim q--> infinity of X(q) 
        #        = 1/(3pi^2) int dk Plin(k)
        #

        qa,lqa,X0,X,Y,plin = self._getXYZ(k0)

        def kernel(k,n):
            val = 4*np.pi * np.exp(-0.5*k**2*(X(lqa)+Y(lqa))) 
            if n==0:
                val -=  4*np.pi * np.exp(-0.5*k**2*X0) # subtract off contribution from all scales
            else:
                val *= (k**2*Y(lqa)/(k*qa))**n
            return val

        nk = len(kv)
        i=0
        pzel=np.zeros((nk,N+2))
        for k in kv:
            val = 0.0
            ss = sbt(qa,kmax=k)

            for n in range(0,N+1):
                dp = kernel(k,n)
                pzeln = ss.run(dp, direction=1, norm=False, l=n)[-1]
                pzel[i,n+1] = pzeln
                val += pzel[i,n+1]
            pzel[i,0] = val
            if (i+1)%10 == 0 : print(f"k0 = {k0:0.2f} ; wavenumber {i+1} of {nk}     ",end='\r')
            i += 1
        return pzel,plin(np.log(kv))

