import numpy as np
from scipy import optimize
from scipy.constants import mu_0, epsilon_0
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline,splrep, BSpline
from scipy.sparse import csr_matrix, csc_matrix
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import lu_factor, lu_solve
from scipy import signal
import  os
eps= np.finfo(float).eps
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod

class TorchHelper:
    @staticmethod
    def to_tensor_r(x, dtype=torch.float32, device='cpu'):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def to_tensor_c(x, dtype=torch.complex64, device='cpu'):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)

class Pelton_res_f():
    def __init__(self, freq=None, con=False,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.clim = TorchHelper.to_tensor_r(clim)
        self.con = con

    def f(self,p):
        """
        Return the Pelton resistivity model for PyTorch auto-differentiation.

        **Parameters**
        - `p` (`torch.Tensor`): Model parameters
            - `p[0]` : log(res0)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c

        **Model Equation**

        ```math
        \rho(\omega) = \rho_0 \left[ 1 - \eta \left(1 - \frac{1}{1+(i\omega\tau)^c} \right) \right]
        = \rho_0 \left[ \frac{\tau^{-c} + (1-\eta)(i\omega)^c}{\tau^{-c} + (i\omega)^c} \right]
        ```
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        if self.con:
            return torch.exp(-p[0])/(tc +(1.0-p[1])*iwc)*(tc+iwc)
        else:
            return torch.exp(p[0])*(tc +(1.0-p[1])*iwc)/(tc+iwc)      

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = 2
        ind_c = 3
       
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton(Pelton_res_f):
    """Alias for Pelton_res_f with a simplified name."""
    pass

class ColeCole_f():
    def __init__(self, freq=None, res=False,
            conlim= [1e-5,1e2],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq =  TorchHelper.to_tensor_c(freq)if freq is not None else None
        self.res = res
        self.reslim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.clim = TorchHelper.to_tensor_r(clim)

    def f(self,p):
        """
        Return the Pelton resistivity model for PyTorch auto-differentiation.

        **Parameters**
        - `p` (`torch.Tensor`): Model parameters
            - `p[0]` : log(con8)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c

        **Model Equation**

        ```math
        \sigma_\infty \left(1- \dfrac{\eta}{1+(i\omega \tau)^c}\right)
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        if self.res:
            return torch.exp(-p[0])/((1.0-p[1])*tc +iwc)*(tc+iwc)
        else:
            return torch.exp( p[0])*((1.0-p[1])*tc +iwc)/(tc+iwc)
        
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()       
        mvec_tmp[0] = torch.clamp(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1] = torch.clamp(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = torch.clamp(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = torch.clamp(mvec[3], self.clim.min(), self.clim.max())
        return mvec_tmp

class ColeCole(ColeCole_f):
    """Alias for ColeCole_f with a simplified name."""
    pass

class Debye_sum_f():
    def __init__(self,
            freq=None,
            times=None, tstep=None, con=False,
            taus=None,
            reslim= [1e-2,1e5],
            chglim= [0, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.taus = TorchHelper.to_tensor_c(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.con = con
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def f(self, p):
        """
        Pelton linear weighted Debye model in frequency domain.
        Parameters:
            p[0]: log(rho0)
            p[1:1+ntau]: etas (relaxation weights)
        Returns:
            Complex-valued resistivity at each frequency.
        ```Math
        \rho(\omega)=\rho_0 \left[ 1 -\sum_{j=1}^n \eta_j + \sum_{j=1}^n \dfrac{\eta_j}{1+i\omega\tau_j}\right]
        """
        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau].to(dtype=torch.cfloat)
        omega = 2.0 * torch.pi * self.freq
        omega = omega.view(-1, 1)  # shape: [nfreq, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        iwt = 1.0j * omega * taus  # shape: [nfreq, ntau]
        term =  etas / (1.0 + iwt)  # shape: [nfreq, ntau]
        if self.con:
            return 1.0/rho0 / (1.0 -etas.sum(dim=1)+ term.sum(dim=1))
        else:
            return rho0 * (1.0 -etas.sum(dim=1)+ term.sum(dim=1))  # shape: [nfreq]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Debye_Sum_Ser_f(Debye_sum_f):
    """Alias for Debye_sum_f with a specific name."""
    pass

class Debye_sum_t():
    def __init__(self,
            times=None, tstep=None, taus=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        assert np.all(times >= -eps), "Times must be greater than or equal to 0"
        if len(times) > 1:
            assert np.all(np.diff(times) >= -eps), "Time values must be in ascending order."
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.taus = TorchHelper.to_tensor_r(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.reslim = TorchHelper.to_tensor_r(np.log(reslim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def t(self, p, tstep=None):
        """
        Time-Domain Debye-Combination Series Configuration in Resistivity form.
        Parameters:
            p[0]: log(rho0)
            p[1:1+ntau]: etas (relaxation weights)
        Returns:
            Real value resistivity given times.
        ```Math
        \rho(t)=\rho_0 \left[ \left(1 -\sum_{j=1}^n \eta_j \right) \delta(t)+ \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
        """
        if tstep is not None:
            self.tstep = TorchHelper.to_tensor_r(tstep)

        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau]
        ind_0 = torch.where(self.times == 0)[0]
        times = self.times.view(-1, 1)  # shape: [ntime, 1]
        # taus = taus.view(1, -1)  # shape: [1, ntau]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        term = etas/taus*torch.exp(-times/taus)  # shape: [ntime, ntau]
        term_sum = term.sum(dim=1)  # shape: [ntime]
        term_sum[ind_0] = 1.0-etas.sum(dim=1)  # Set the value at t=0 to 1 - sum(etas)
        if self.tstep is not None:
            ind_pos = torch.where(self.times > 0)
            term_sum[ind_pos] *= self.tstep
            return rho0 * (term_sum)  # shape: [tau]
        else:
            return rho0 * (term_sum)  # shape: [tau]
 
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        # if self.taus is None:
        #     ind_tau = 1+self.ntau+np.arange(self.ntau)
        #     mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        #     mvec_tmp[ind_tau[0]] = torch.clamp(mvec[ind_tau[0]],
        #                 self.taulim.min()+self.taulimspc.min(), self.taulim.min() + self.taulimspc.max())
        #     a_local = torch.tensor(np.r_[-1.0,1.0], dtype=torch.float)
        #     for i in range(self.ntau-1):
        #         mvec_tmp[ind_tau[i:i+2]] = self.proj_halfspace(
        #             mvec[i:i+2], -a_local, self.taulimspc.max())
        #         mvec_tmp[ind_tau[i:i+2]] = self.proj_halfspace(
        #             mvec[i:i+2],  a_local, self.taulimspc.min())
        #     mvec_tmp[ind_tau[-1]] = torch.clamp(mvec[ind_tau[-1]],
        #                 self.taulim.max()-self.taulimspc.max(), self.taulim.min() - self.taulimspc.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Debye_Sum_Ser_t(Debye_sum_t):
    """Alias for Debye_sum_t with a specific name."""
    pass

class Debye_Sum_Par_f():
    def __init__(self,
            freq=None, taus=None, res=False,
            conlim= [1e-5,1e2],
            chglim= [0, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.freq = TorchHelper.to_tensor_c(freq) if freq is not None else None
        self.taus = TorchHelper.to_tensor_c(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.res= res
        self.conlim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def f(self, p):
        """
        Frequency domain Debye-Combination model Parallel configuration in Conductivity form.
        Parameters:
            p[0]: log(con8)
            p[1:1+ntau]: etas (relaxation weights)
        Returns:
            Complex-valued conductivity at each frequency.
        ```Math
        \sigma(\omega)=\sigma_\infty\left(1- \sum_{j=1}^n\dfrac{\eta_j}{1+i\omega\tau_j}\right)
        """
        con8 = torch.exp(p[0])
        etas = p[1:1 + self.ntau].to(dtype=torch.cfloat)
        omega = 2.0 * torch.pi * self.freq
        omega = omega.view(-1, 1)  # shape: [nfreq, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        iwt = 1.0j * omega * taus  # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt)  # shape: [nfreq, ntau]
        if self.res:
            return 1.0/con8 / (1.0 - term.sum(dim=1))
        else:
            return con8 * (1.0 - term.sum(dim=1))  # shape: [nfreq]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_con = 0
        ind_chg = 1+np.arange(self.ntau)
        mvec_tmp[ind_con] = torch.clamp(mvec[ind_con], self.conlim.min(), self.conlim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Debye_Sum_Par_t():
    def __init__(self,
            times=None, tstep=None, taus=None,
            conlim= [1e-5,1e3],
            chglim= [1e-3, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        assert np.all(times >= -eps), "Times must be greater than or equal to 0"
        if len(times) > 1:
            assert np.all(np.diff(times) >= -eps), "Time values must be in ascending order."
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.tstep = TorchHelper.to_tensor_r(tstep) if tstep is not None else None
        self.taus = TorchHelper.to_tensor_r(taus) if taus is not None else None
        self.ntau = len(taus) if taus is not None else None
        self.conlim = TorchHelper.to_tensor_r(np.log(conlim))
        self.chglim = TorchHelper.to_tensor_r(chglim)
        self.taulim = TorchHelper.to_tensor_r(np.log(taulim))
        self.taulimspc = TorchHelper.to_tensor_r(np.log(taulimspc))

    def t(self, p, tstep=None):
        """
        Time-Domain Debye-Combination Parallel Configuration in Conductivity form.
        Parameters:
            p[0]: log(con8)
            p[1:1+ntau]: etas (relaxation weights)
        Returns:
            Real value conductivity given times.
        ```Math
        \sigma(t)=\sigma_\infty \left[ \delta(t)- \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
        """
        if tstep is not None:
            self.tstep = TorchHelper.to_tensor_r(tstep)

        con8 = torch.exp(p[0])
        etas = p[1:1 + self.ntau]
        ind_0 = torch.where(self.times == 0)[0]
        times = self.times.view(-1, 1)  # shape: [ntime, 1]
        taus = self.taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        term = etas/taus*torch.exp(-times/taus)  # shape: [ntime, ntau]
        term_sum = term.sum(dim=1)  # shape: [ntime]
        term_sum[ind_0] = 1  # Set the value at t=0 to 1 - sum(etas)
        if self.tstep is not None:
            ind_pos = torch.where(self.times > 0)
            term_sum[ind_pos] *= self.tstep
        return con8 * (term_sum)  # shape: [tau]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_con = 0
        ind_chg = 1+np.arange(self.ntau)
    
        mvec_tmp[ind_con] = torch.clamp(mvec[ind_con], self.conlim.min(), self.conlim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x

class Pelton_res_f_two():
    def __init__(self, freq=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = torch.tensor(freq, dtype=torch.cfloat) if freq is not None else None
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)
    
    def f(self,p):
        """
        Pelton two relaxation resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta
        p[2] : log(tau1)
        p[3] : c1
        p[4] : log(tau2)
        p[5] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[5]
        tc2 = torch.exp(-p[4]*p[5])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)*(tc2)/(tc1+iwc1)/(tc2+iwc2)
        return f
    
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = [2,4]
        ind_c = [3,5] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton_res_f_dual():
    def __init__(self, freq=None,
        reslim= [1e-2,1e5],
        chglim= [1e-3, 0.9],
        taulim= [1e-8, 1e4],
        clim= [0.2, 0.8]
            ):
        self.freq = torch.tensor(freq, dtype=torch.cfloat) if freq is not None else None
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def f(self,p):
        """
        Pelton dual cole-cole resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta1
        p[2] : log(tau1)
        p[3] : c1
        p[4] : eta2
        p[5] : log(tau2)
        p[6] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[6]
        tc2 = torch.exp(-p[5]*p[6])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)/(tc1+iwc1)*(tc2 +(1.0-p[4])*iwc2)/(tc2+iwc2)
        return f

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = [1,4]
        ind_tau = [2,5]
        ind_c = [3,6] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class BaseSimulation:
    eps = torch.finfo(torch.float32).eps
    @abstractmethod
    def dpred(self,m):
        pass
    @abstractmethod
    def J(self,m):
        pass
    @abstractmethod
    def project_convex_set(self,m):
        pass

class InducedPolarizationSimulation(BaseSimulation):
    AVAILABLE_MODES = ['tdip_t', 'tdip_f', 'sip_t', 'sip']

    eps = torch.finfo(torch.float32).eps
    def __init__(self, 
                 ip_model=None,
                 mode=None,
                 times=None,
                 basefreq=None,
                 window_mat=None,
                 windows_strt=None,
                 windows_end=None,
                 log2min=-6,
                 log2max=6,
                 ):
        """
        Induced Polarization Simulation.

        Parameters:
        - ip_model: input IP model for simulation
        - mode (str): One of ['tdip_t', 'tdip_f', 'sip_t', 'sip']
        - times: Time values (1D list or tensor)
        - basefreq: Frequency base (1D list or tensor)
        - window_mat: Matrix used for windowing the response
        - windows_strt, windows_end: (Optional) Start and end times for windows
        """
        #  Validate the mode
        if mode is not None and mode not in self.AVAILABLE_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.AVAILABLE_MODES}")

        self.ip_model = ip_model
        self.mode=mode
        self.times = TorchHelper.to_tensor_r(times) if times is not None else None
        self.basefreq = TorchHelper.to_tensor_r(basefreq) if basefreq is not None else None
        self.window_mat = TorchHelper.to_tensor_r(window_mat) if window_mat is not None else None 
        self.windows_strt = TorchHelper.to_tensor_r(windows_strt) if windows_strt is not None else None
        self.windows_end = TorchHelper.to_tensor_r(windows_end) if windows_end is not None else None
        self.windows_width = TorchHelper.to_tensor_r(windows_end-windows_strt) if windows_strt is not None else None
        self.log2min = log2min
        self.log2max = log2max
   
    def count_data_windows(self, times):
        nwindows = len(self.windows_strt)
        count_data = torch.zeros(nwindows)
        for i in torch.arange(nwindows):
            start = self.windows_strt[i]
            end = self.windows_end[i]
            ind_time = (times >= start) & (times <= end)
            count_data[i] = ind_time.sum()
        return count_data

    def get_freq_windowmat(self,tau, max=22, ign_0=False):
        log2max = self.log2max
        log2min = self.log2min
        freqend = ((1 / tau) * 2 ** log2max).item()
        freqstep = ((1 / tau) * 2 ** log2min).item()
        freq = torch.arange(0, freqend, freqstep)
        times = torch.arange(0, 1/freqstep,1/freqend)

        count = self.count_data_windows(times)
        # while count[self.windows_end==self.windows_end.max()] <2:
        #     # print(count)
        #     log2min -= 1
        #     freqstep = ((1/tau)*2**log2min).item()
        #     if log2max-log2min >= max:
        #         print('some windows are too narrow')
        #         break
        #     freq = torch.arange(0,freqend,freqstep)
        #     times = torch.arange(0, 1/freqstep,1/freqend)
        #     count = self.count_data_windows(times)

        while times.max() < self.windows_end.max():
            # print(count)
            log2min -= 1
            freqstep = ((1/tau)*2**log2min).item()
            if log2max-log2min >= max:
                print('some windows are too narrow')
                break
            freq = torch.arange(0,freqend,freqstep)
            times = torch.arange(0, 1/freqstep,1/freqend)
            # freq = torch.fft.fftfreq(len(freq), d=1/freqend)
            count = self.count_data_windows(times)
        
        ind_narrow = self.windows_width == self.windows_width.min()
        if ind_narrow.sum() >= 2:
            # print(torch.where(ind_narrow)[0])
            # print(torch.where(ind_narrow)[0][0].item())
            ind_narrow = torch.where(ind_narrow)[0][0].item()

        while count[ind_narrow] <2:
            # print(count)
            log2max += 1
            freqend = ((1/tau)*2**log2max).item()
            if log2max-log2min >= max:
                print('some windows are too narrow')
                break
            freq = torch.arange(0,freqend,freqstep)
            times = torch.arange(0, 1/freqstep,1/freqend)
            # freq = torch.fft.fftfreq(len(times), d=1/freqend)
            count = self.count_data_windows(times)

        self.times = times
        # freq = torch.fft.fftfreq(len(freq), d=1/freqend)

        self.ip_model.freq = freq
        self.get_window_matrix(times=times)
        
    def freq_symmetric(self,f):
        f_sym = torch.zeros_like(f, dtype=torch.cfloat)
        nfreq = len(f)
        if nfreq  %2 == 0:
            nfreq2 = nfreq // 2
            f_sym[:nfreq2] = f[:nfreq2]
            f_sym[nfreq2] =f[nfreq2].real
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2].conj(), dims=[0])
        # Ensure symmetry at the Nyquist frequency (if even length)
        else:
            nfreq2 = nfreq // 2 
            f_sym[:nfreq2+1] = f[:nfreq2+1]
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2+1].conj(), dims=[0])
        return f_sym

    def compute_fft(self,f):
        f_sym = self.freq_symmetric(f)
        t = torch.fft.ifft(f_sym) 
        return t

    def get_windows(self,windows_cen):
        # windows_cen = TorchHelper.to_tensor_r(windows_cen[windows_cen>0])
        windows_cen = TorchHelper.to_tensor_r(windows_cen)
        windows_strt = torch.zeros_like(windows_cen)
        windows_end  = torch.zeros_like(windows_cen)
        dt = torch.diff(windows_cen)
        windows_strt[1:] = windows_cen[:-1] + dt / 2
        windows_end[:-1] = windows_cen[1:] - dt / 2
        # windows_strt[0] = 10*eps
        windows_strt[0] = windows_cen[0] - dt[0] / 2
        windows_end[-1] = windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        self.windows_width = windows_end - windows_strt

    def get_window_matrix(self, times=None, sum=False):
        if times is not None:
            self.times = TorchHelper.to_tensor_r(times)
        nwindows = len(self.windows_strt)
        window_matrix = torch.zeros((nwindows, len(self.times)))
        for i in range(nwindows):
            ind_time = (self.times >= self.windows_strt[i]) & (self.times <= self.windows_end[i])
            count = ind_time.sum()
            if count > 0:
                if sum:
                    window_matrix[i, ind_time] = 1.0
                else:
                    window_matrix[i, ind_time] = 1.0 / count
            else:
                print(f"Warning: No data points found for window {i+1} ({self.windows_strt[i]} to {self.windows_end[i]})")
        self.window_mat = window_matrix
        # window_matrix = torch.zeros((nwindows+1, len(self.times)))
        # window_matrix[0,0] = 1
        # for i in range(nwindows):
        #     ind_time = (self.times >= self.windows_strt[i]) & (self.times <= self.windows_end[i])
        #     if ind_time.sum() > 0:
        #         if sum:
        #             window_matrix[i+1, ind_time] = torch.ones(ind_time.sum())
        #         window_matrix[i+1, ind_time] = 1.0/(ind_time.sum())
        # self.window_mat = window_matrix

    def set_current_wave(self, basefreq=None, curr_duty=0.5):
        """
        Set the current wave for the simulation as rectangular wave form.
        """
        if basefreq is not None:
            self.basefreq = basefreq
        curr =  0.5*(1.0+signal.square(2*np.pi*(self.basefreq*self.times),duty=curr_duty))
        self.curr = TorchHelper.to_tensor_r(curr)
        self.curr_duty=  curr_duty

    def get_windows_matrix_curr(self, smp_freq, nlin, nlin_strt, basefreq=None) :
        """
        Get the windows matrix based on currenqt wave form.
        """
        if basefreq is not None:
            self.basefreq = basefreq
        rec_time = 1/self.basefreq
        time_step = 1/smp_freq
        windows_lin = (np.arange(nlin)+nlin_strt)*time_step 
        windows_log_strt = (nlin+nlin_strt)*time_step
        logstep = np.log10(windows_lin[-1]/windows_lin[-2])
        windows_log_end = self.curr_duty/self.basefreq
        windows_log = 10**np.arange(
            np.log10(windows_log_strt),
            np.log10(windows_log_end), logstep)
        windows_cen=np.r_[windows_lin, windows_log]
        nhalf = len(windows_cen)
        windows_cen = np.r_[windows_cen, self.curr_duty/self.basefreq + windows_cen]
        self.get_windows(windows_cen)

        self.windows_end[nhalf-1]  = self.curr_duty/self.basefreq
        self.windows_strt[nhalf] = self.windows_strt[0] + self.curr_duty/self.basefreq
        self.windows_end[-1] = 1/self.basefreq
        self.get_window_matrix(times=self.times)

    def fft_convolve(self, d, f):
        """
        Perform 1D linear convolution using FFT (like scipy.signal.fftconvolve).
        Assumes x and h are 1D tensors.
        Returns output of length (len(x) + len(h) - 1)
        """
        nd = d.shape[0] 
        nf = f.shape[0] 
        # Compute FFTs (real FFT for speed)
        D = torch.fft.rfft(d, n=nd+nf-1)
        F = torch.fft.rfft(f, n=nd+nf-1)
        # Element-wise multiplication
        DF = D * F
        # Inverse FFT to get back to time domain
        return  torch.fft.irfft(DF, n=nd+nf-1)
 
    def dpred(self,m):
        if self.mode=="tdip_t" :
            ip_t=self.ip_model.t(m)
            volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
            return self.window_mat@volt

        if self.mode=="tdip_f":
            ip_f = self.ip_model.f(m)
            ip_t = torch.fft.ifft(ip_f).real
            volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
            return self.window_mat@volt

        # if self.mode=="tdip_f":
        #     self.get_freq_windowmat(tau=torch.exp(m[2]))
        #     self.set_current_wave()
        #     ip_f = self.ip_model.f(m)
        #     ip_fsym = self.freq_symmetric(ip_f)
        #     ip_t = torch.fft.ifft(ip_fsym).real
        #     volt = self.fft_convolve(ip_t,self.curr)[: len(self.times)]
        #     return self.window_mat@volt
            # assert len(self.times) == len(self.ip_model.freq)
            # curr_f = torch.fft.fft(self.curr)
            # volt = torch.fft.ifft(ip_fsym*curr_f).real
            # volt = volt[: len(self.times)]
            # return self.window_mat@volt

        if self.mode=="sip_t":
            # self.get_freq_windowmat(tau=torch.exp(m[2]))
            f = self.ip_model.f(m)
            t = self.compute_fft(f)/(self.times[1]-self.times[0])
            t_real = t.real
            return self.window_mat@t_real
        
        if self.mode=="sip":
            f = self.ip_model.f(m)
            f_real = f.real
            f_imag = f.imag
            if self.window_mat is None:
                return torch.cat([f_real, f_imag])
            else:
                return torch.cat([self.window_mat@f_real, self.window_mat@f_imag])

    def J(self,m):
        return torch.autograd.functional.jacobian(self.dpred, m)   

    def J_prd(self,J):
        """"
        Retursn numpy arrays given the Jacobian matrix J in numpy format.
        1. J_pro: the projection of the eta vectors on the resistivity vector, normalized
        2. etas_norm: the norm of the eta vectors, normalized by the norm of Jacobian matrix
        """
        J_0 = J[:,0]
        J_0_norm = np.linalg.norm(J_0)
        J_prd = np.zeros(J.shape[1]-1)
        for i in range(J.shape[1]-1):
            J_i = J[:,i+1]
            J_i_norm = np.linalg.norm(J_i)
            J_prd[i] = np.dot(J_0, J_i)/J_i_norm/ J_0_norm
        return J_prd

    def Jvec(self, m, v):
        return torch.autograd.functional.jvp(self.dpred, m, v)

    def Jtvec(self, m, v):
        return torch.autograd.functional.vjp(self.dpred, m, v)

    def project_convex_set(self,m):
        return self.ip_model.clip_model(m).detach().requires_grad_(False)

class Optimization():  # Inherits from BaseSimulation
    def __init__(self,
                sim,
                dobs=None,
                Wd=None, Ws=None, Wx=None, alphas=1, alphax=1,
                Ws_threshold=1e-12
                ):
        self.sim = sim  # Composition: opt_tmp has a simulation
        self.dobs= dobs
        self.Wd = Wd
        self.Ws = Ws
        self.Ws_threshold = Ws_threshold
        self.Wx = Wx
        self.alphas = alphas
        self.alphax = alphax
    
    def dpred(self, m):
        return self.sim.dpred(m)  # Calls InducedPolarization's dpred()

    def J(self, m):
        return self.sim.J(m)  # Calls InducedPolarization's J()
    
    def Jvec(self, m, v):
        return self.sim.Jvec(m, v)
    
    def Jtvec(self, m, v):
        return self.sim.Jtvec(m, v)
    
    def project_convex_set(self,m):
        return self.sim.project_convex_set(m)

    def get_Wd(self,ratio=0.10, plateau=0, sparse=False):
        dobs_clone = self.dobs.clone().detach()
        noise_floor = plateau * torch.ones_like(dobs_clone)
        std = torch.sqrt(noise_floor**2 + (ratio * torch.abs(dobs_clone))**2)
        self.Wd =torch.diag(1 / std.flatten())
        if sparse:
            self.Wd = self.Wd.to_sparse()

    def get_Ws(self, mvec, sparse=False):
        self.Ws = torch.eye(mvec.shape[0])
        if sparse:
            self.Ws = self.Ws.to_sparse()
    
    def loss_func_L2(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * torch.sum(r**2)
        phim = 0

        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = 0.5 * self.alphas*torch.sum(rms**2)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*torch.sum(rmx**2) 
        return phid+beta*phim, phid, phim 

    def loss_func_L1reg(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * torch.sum(r**2)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = self.alphas*torch.sum(abs(rms))
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(abs(rmx)) 
        return phid+beta*phim, phid, phim 

    def BetaEstimate_byEig(self,m, beta0_ratio=1.0, 
                eig_tol=eps,l1reg=False, norm=True,update_Wsen=False):
        mvec = m.clone().detach()
        J = self.J(mvec)

        if update_Wsen:
            self.update_Ws(J)    

        # Prj_m = self.Prj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data = 0.5* J.T @ self.Wd.T @ self.Wd @ J 
        
        # Effective regularization term with projection matrix
        # A_reg = alphax* Prj_m.T @ Wx.T @ Wx @ Prj_m
        A_reg = torch.zeros_like(A_data)
        if self.Wx is not None:
            if l1reg:
                diag = torch.diag(self.Wx.T @ self.Wx)
                A_reg += self.alphax *torch.diag(diag**0.5)
            else:
                A_reg += 0.5*self.alphax * (self.Wx.T @ self.Wx)
        if self.Ws is not None:
            if l1reg:
                A_reg += self.alphas * self.Ws
            else:
                A_reg += 0.5*self.alphas * (self.Ws.T @ self.Ws)

        if norm:
            lambda_d = torch.linalg.norm(A_data, ord=2)  # Spectral norm ≈ largest eigval
            lambda_r = torch.linalg.norm(A_reg, ord=-2)  # Smallest eigval approx (not accurate, but fast)
        else:
            eig_data = torch.linalg.eigvalsh(A_data)
            eig_reg = torch.linalg.eigvalsh(A_reg)
            
            # Ensure numerical stability (avoid dividing by zero)
            eig_data = eig_data[eig_data > eig_tol]
            eig_reg = eig_reg[eig_reg > eig_tol]

            # Use the ratio of eigenvalues to set beta range
            lambda_d = torch.max(eig_data)
            lambda_r = torch.min(eig_reg)
        return beta0_ratio * lambda_d / lambda_r
  
    def compute_sensitivity(self,J):
        return  torch.sqrt(torch.sum(J**2, axis=0))

    def update_Ws(self, J):
        Sensitivity = self.compute_sensitivity(self.Wd@J)
        Sensitivity /= Sensitivity.max()
        Sensitivity = np.clip(Sensitivity, self.Ws_threshold, 1)
        self.Ws = torch.diag(Sensitivity)

    # def update_Ws(self, m):
    #     """Approximate sensitivity using Jᵀ Wᵀ W J diag only (no full Jacobian)"""
    #     m = m.detach().clone().requires_grad_(True)
    #     nparam = m.numel()
    #     sensitivity_sq = torch.zeros(nparam)

    #     for i in range(nparam):
    #         # Basis vector ei
    #         ei = torch.zeros_like(m)
    #         ei[i] = 1.0
    #         # Compute J @ ei = directional derivative w.r.t. m_i
    #         dpred, dFdm_i = self.Jvec(m, ei)
    #         wr = self.Wd @ dFdm_i  # Wd @ column of J
    #         sensitivity_sq[i] = torch.sum(wr ** 2)
    #     self.Ws= torch.sqrt(sensitivity_sq)
    #     return self.Ws

    def GradientDescent(self,mvec_init, niter, beta0, l1reg=False, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4,ELS=True, BLS=True ):

        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        mvec_old = mvec_init.detach().clone().requires_grad_(True)
        m_ref = mvec_old.detach()
        beta= beta0

        for i in range(niter):
            beta =  beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))

            if update_Wsen:
                J = self.J(mvec_old)
                self.update_Ws(J)
            if l1reg:
                f_old, phid, phim = self.loss_func_L1reg(mvec_old,beta=beta,m_ref=m_ref)
            else:
                f_old, phid, phim = self.loss_func_L2(mvec_old,beta=beta,m_ref=m_ref) 
            f_old.backward()   # Compute the gradient of f_old

            # if mvec_old.grad is not None:
            #     mvec_old.grad.zero_()  # Zero out the gradient before computing it  
            g = mvec_old.grad  # Get the gradient of mvec_old

            self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_old.detach().numpy())
            self.data_prg.append(self.dpred(mvec_old).detach().numpy())

           # Exact line search
            if ELS:
                dpred, Jg =self.Jvec(mvec_init,g)
                t = torch.sum(g**2)/torch.sum((self.Wd @ Jg )**2)
            else:
                t = 1.

            g_norm = torch.linalg.norm(g, ord=2)
            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            dm = t*g.flatten()  # Ensure dm is a 1D tensor
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            if l1reg:
                f_new, phid, phim = self.loss_func_L1reg(mvec_new, beta=beta,m_ref=m_ref)
            else:
                f_new, phid, phim = self.loss_func_L2(mvec_new, beta=beta,m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            if BLS:
                while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                    s *= torch.tensor(sfac)
                    mvec_new = self.project_convex_set(mvec_old - s * dm)
                    if l1reg:
                        f_new, phid, phim = self.loss_func_L1reg(mvec_new,beta=beta,m_ref=m_ref)
                    else:
                        f_new, phid, phim = self.loss_func_L2(mvec_new,beta=beta,m_ref=m_ref) 
                    if s < torch.tensor(stol):
                        break
            mvec_old = mvec_new.detach().clone().requires_grad_(True)
            if print_update:
                print(f'{i + 1:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e},  f:{f_new:.1e}')

        self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_new.detach().numpy())
        self.data_prg.append(self.dpred(mvec_new).detach().numpy())
        return mvec_new

    def GaussNewton(self,mvec_init, niter, beta0, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4):
        mvec_old = mvec_init #.detach()
        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        m_ref = mvec_init.detach()
        f_old, phid, phim = self.loss_func_L2(mvec_old, beta=beta0, m_ref=m_ref)
        self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_old.detach().numpy())
        self.data_prg.append(self.dpred(mvec_old).detach().numpy())

        for i in range(niter):
            beta = beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))
            rd = self.Wd@(self.dpred(mvec_old) - self.dobs)
            J = self.J(mvec_old)
            if update_Wsen:
                self.update_Ws(J)                
            g = J.T @ self.Wd.T@ rd
            H = J.T @ self.Wd.T@ self.Wd@J
            if m_ref is not None:
                g += beta * self.alphas * (self.Ws.T@self.Ws@ (mvec_old - m_ref))
                H += beta * self.alphas * self.Ws.T@self.Ws
            if self.Wx is not None:
                g += beta * self.alphax * (
                    self.Wx.T @ self.Wx @ mvec_old)
                H += beta * self.alphax * self.Wx.T @ self.Wx
            
            dm = torch.linalg.solve(H, g).flatten()  # Ensure dm is a 1D tensor
            g_norm = torch.linalg.norm(g, ord=2)

            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            f_new, phid, phim = self.loss_func_L2(mvec_new, beta=beta,m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                s *= torch.tensor(sfac)
                mvec_new = self.project_convex_set(mvec_old - s * dm)
                f_new, phid, phim = self.loss_func_L2(mvec_new,beta=beta,m_ref=m_ref) 
                if s < torch.tensor(stol):
                    break
            mvec_old = mvec_new
            f_old = f_new
            self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_new.detach().numpy())
            self.data_prg.append(self.dpred(mvec_new).detach().numpy())
            if print_update:
                print(f'{i + 1:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e},  f:{f_new:.1e}')
        return mvec_new
    