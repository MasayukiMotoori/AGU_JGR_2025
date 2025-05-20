import numpy as np
from scipy import optimize
from scipy.constants import mu_0, epsilon_0
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial, roots_legendre, eval_legendre
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline,splrep, BSpline
from scipy.sparse import csr_matrix, csc_matrix
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import lu_factor, lu_solve
from scipy import signal
import empymod
import discretize
import  os
from abc import ABC, abstractmethod
eps= np.finfo(float).eps

class TikonovInversion():
    def __init__(self, G_f, Wd, alphax=1.,Wx=None,
        alphas=1., Ws=None, m_ref=None,Proj_m=None,m_fix=None,
        sparse_matrix=False
        ):  
        self.G_f = G_f
        self.Wd = Wd
        self.Wx = Wx
        self.Ws = Ws
        self.nD = G_f.shape[0]
        self.nP = G_f.shape[1]
        self.alphax = alphax
        self.Proj_m = Proj_m  
        self.m_fix = m_fix
        if Proj_m is not None:
            assert Proj_m.shape[0] == self.nP
            self.nM = Proj_m.shape[1]
        else:
            self.Proj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP)
        self.alphas = alphas
        self.m_ref=m_ref

        self.sparse_matrix = sparse_matrix
    
    def get_Wx(self):
        nP = self.nP
        Wx = np.zeros((nP-1, nP))
        element = np.ones(nP-1)
        Wx[:,:-1] = np.diag(element)
        Wx[:,1:] += np.diag(-element)
        self.Wx = Wx
        return Wx
    
    def get_Ws(self):
        nM = self.nM
        Ws = np.eye(nM)
        self.Ws=  Ws
        return Ws

    def recover_model(self, dobs, beta, sparse_matrix=False):
        # This is for the mapping 
        G_f = self.G_f
        Wd = self.Wd
        alphax = self.alphax
        alphas = self.alphas
        Wx = self.Wx
        Ws = self.Ws
        m_ref= self.m_ref
        Proj_m = self.Proj_m
        m_fix= self.m_fix
        sparse_matrix = self.sparse_matrix
        
        left = Proj_m.T @G_f.T @ Wd.T @ Wd @ G_f@Proj_m
        left += beta * alphax * (Proj_m.T @Wx.T @ Wx@Proj_m) 
        if m_ref is not None:
            left += beta * alphas * (Ws.T @ Ws)
        if sparse_matrix:
            left = csr_matrix(left)
        right =   G_f.T @ Wd.T @Wd@ dobs@Proj_m
        right += -m_fix.T@G_f.T@Wd.T@Wd@G_f@Proj_m
        right+= -beta*alphax* m_fix.T@Wx.T@Wx@Proj_m
        if m_ref is not None:
            right+= beta*alphas*m_ref.T@Ws.T@Ws
        m_rec = np.linalg.solve(left, right)
        #filt_curr = spsolve(left, right)
        rd = Wd@(G_f@Proj_m@m_rec-dobs)
        rmx = alphax*Wx@Proj_m@m_rec
        if m_ref is not None:
            rms = alphas*Ws@(m_rec-m_ref)

        phid = 0.5 * np.dot(rd, rd)
        phim = 0.5 * np.dot(rmx,rmx)
        if m_ref is not None:
            phim+=0.5 * np.dot(rms,rms)
        p_rec = m_fix + Proj_m@m_rec
        return p_rec, phid, phim
    
    def tikonov_inversion(self,beta_values, dobs):
        n_beta = len(beta_values)
        nP= self.nP

        mrec_tik = np.zeros(nP, n_beta)  # np.nan * np.ones(shape)
        phid_tik = np.zeros(n_beta)
        phim_tik = np.zeros(n_beta) 
        for i, beta in enumerate(beta_values): 
            mrec_tik[:, i], phid_tik[i], phim_tik[i] = self.recover_model(
            dobs=dobs, beta=beta)
        return mrec_tik, phid_tik, phim_tik

    
    def estimate_beta_range(self, num=20, eig_tol=1e-12):
        G_f = self.G_f
        alphax=self.alphax
        alphas=self.alphas
       
        Wd = self.Wd
        Wx = self.Wx
        Ws= self.Ws
        Proj_m = self.Proj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data = Proj_m.T @ G_f.T @ Wd.T @ Wd @ G_f @ Proj_m
        eig_data = np.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        A_reg = alphax* Proj_m.T @ Wx.T @ Wx @ Proj_m
        if Ws is not None:
            A_reg += alphas * (Ws.T @ Ws)
        eig_reg = np.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > eig_tol]
        eig_reg = eig_reg[eig_reg > eig_tol]

        # Use the ratio of eigenvalues to set beta range
        beta_min = np.min(eig_data) / np.max(eig_reg)
        beta_max = np.max(eig_data) / np.min(eig_reg)
        
        # Generate 20 logarithmically spaced beta values
        beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num=num)
        return beta_values

class projection_convex_set:
    def __init__(self,maxiter=100, tol=1e-2,
        lower_bound=None, upper_bound=None, a=None, b=None):
        self.maxiter = maxiter
        self.tol = tol
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = a
        self.b = b

    def get_param(self, param, default):
        return param if param is not None else default
        
    def projection_halfspace(self, a, x, b):
        a = self.get_param(a, self.a)
        b = self.get_param(b, self.b)
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def projection_plane(self, a, x, b):
        a = self.get_param(a, self.a)
        b = self.get_param(b, self.b)
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a))
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def clip_model(self, x, lower_bound=None, upper_bound=None):
        lower_bound = self.get_param(lower_bound, self.lower_bound)
        upper_bound = self.get_param(upper_bound, self.upper_bound)
        clipped_x = np.clip(x, self.lower_bound, self.upper_bound)
        return clipped_x

    def proj_c(self,x, maxiter=100, tol=1e-2):
        "Project model vector to convex set defined by bound information"
        x_c_0 = x.copy()
        x_c_1 = np.zerps_like(x)
        maxiter = self.get_param(maxiter, self.maxiter)
        tol = self.tol
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        a = self.a
        b = self.b
        for i in range(maxiter):
            x_c_1 = self.clip_model(x=x_c_0,lower_bound=lower_bound, upper_bound=upper_bound)
            x_c_1 = self.projection_plane(a=a, x=x_c_1, b=b)
            if np.linalg.norm(x_c_1 - x_c_0) < tol:
                break
            x_c_0 = x_c_1
        return x_c_1

class BaseSimulation:
    @abstractmethod
    def dpred(self,m):
        pass
    @abstractmethod
    def J(self,m):
        pass
    @abstractmethod
    def project_convex_set(self,m):
        pass

class empymod_IP_simulation(BaseSimulation):
    AVAILABLE_MODELS = ['pelton', 'Cole','DDR','DDC']
    def __init__(self, model_base, nlayer, tx_height, m_depth=False,
        nD=0, nlayer_fix=0, Prj_m=None, m_fix=None,
        nM_r = None, nM_m = None, nM_t= None, nM_c=None, nM_d= None,
        recw=None,resmin=1e-3 , resmax=1e6, chgmin=1e-3, chgmax=0.9,
        taumin=1e-6, taumax=1e-1, cmin= 0.4, cmax=0.9,
        taus=None, ip_model='pelton',
        cut_off=None,filt_curr = None,  window_mat = None,
        ):
        if ip_model is not None and ip_model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid ip model '{ip_model}'. Choose from {self.AVAILABLE_MODELS}")
        self.ip_model = ip_model
        self.model_base = model_base
        self.nlayer = int(nlayer)
        self.nlayer_fix = int(nlayer_fix)
        self.m_depth = m_depth
        self.tx_height = tx_height
        self.nP = 4*(nlayer + nlayer_fix)
        self.Prj_m = Prj_m  
        self.m_fix = m_fix
        self.recw = recw
        if Prj_m is not None:
            assert Prj_m.shape[0] == self.nP
            self.nM = Prj_m.shape[1]
        else:
            self.Prj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP) 
            self.nM_r = nlayer
            self.nM_m = nlayer
            self.nM_t = nlayer
            self.nM_c = nlayer
        self.taus = taus
        self.ntau = len(taus) if taus is not None else None
        self.nD = nD
        self.resmin = resmin
        self.resmax = resmax
        self.chgmin = chgmin
        self.chgmax = chgmax
        self.taumin = taumin
        self.taumax = taumax
        self.cmin = cmin
        self.cmax = cmax
        self.cut_off = cut_off
        self.filt_curr = filt_curr
        self.window_mat = window_mat

    def get_param(self, param, default):
        return param if param is not None else default
        
    def get_recBdple(self, tx_side, nquad=3):
        # Get roots and weights for the Legendre quadrature
        roots, weights = roots_legendre(nquad)
       
        # Scale roots by tx_side
        scaled_roots = roots * tx_side / 2.0

        # Create grid positions for x and y
        recx = np.tile(scaled_roots, nquad)  # Repeat scaled_roots nquad times for x
        recy = np.repeat(scaled_roots, nquad)  # Repeat each element of scaled_roots nquad times for y

        # Compute weights matrix (outer product of weights) and normalize
        w_array = weights.reshape(-1, 1)  # Column vector (nquad x 1)
        w_arrayT = weights.reshape(1, -1)  # Row vector (1 x nquad)
        matrix = np.dot(w_array, w_arrayT)  # Outer product to create weights grid
        recw = matrix.reshape(1, -1)  # Flatten the matrix to a 1D row vector
        recw /= recw.sum()  # Normalize weights to ensure they sum to 1
        self.recw = recw
        return recx, recy

    def fix_sea_tau_c(self,
             res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([ # res chg
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m_b =np.block([ # tau c
            [0], # sea water
            [np.ones(nlayer).reshape(-1,1)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, nlayer+2))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_b, np.zeros(nlayer_sum).reshape(-1,1)], # Time constant
        [np.zeros((nlayer_sum,2*nlayer)), np.zeros(nlayer_sum).reshape(-1,1), Prj_m_b], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer) # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = 0
        self.nM_c = 0
        assert self.nP == 4*(nlayer_sum)
        assert self.nM == 2*nlayer
        return Prj_m, m_fix

    def deepsea_signle_layer(self, res_sea, res_base,
         eta_sea=0,  tau_sea=1e-3, c_sea=0.4,
         eta_base=0, tau_base=1e-3, c_base=0.4
         ):
        """
        Three layrer model consits of signle target layer, Sea water and bottom layer.
        The physical property of Sea water and bottom layer
        For Pelton model
        m[0] : resistivity of the layer
        m[1] : chargeability of the layer
        m[2] : time constant of the layer in log
        m[3] : exponent C of the layer
        For Debye Decompostion model
        m[0] : resistivity of the layer
        m[1:ntau+1] : chargeabilities of the layer
        """
        self.nlayer = 1
        nlayer_fix = 2
        # resistivity
        if self.ip_model == 'pelton':
            Prj_m_diag = np.diag(np.r_[
                        0,1,0,  # resistivity 
                        0,1,0, # chargeability
                        0,1,0, # time constant
                        0,1,0, # exponent C
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]
            m_fix = np.r_[np.log(res_sea),0,np.log(res_base), # resistivity
                          eta_sea,   0, eta_base, # chargeability
                        np.log(tau_sea), 0, np.log(tau_base),# time constant
                        c_sea, 0, c_base,# exponent C
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP= Prj_m.shape[0]
            self.nM= Prj_m.shape[1]
            self.nM_r, self.nM_m  = 1,1
            self.nM_t, self.nM_c, self.nM_d = 1,1 , 0
            self.res_sea, self.con_sea = res_sea, 1/res_sea
            self.eta_sea, self.tau_sea, self.c_sea = eta_sea, tau_sea, c_sea

        if self.ip_model == 'Cole':
            Prj_m_diag = np.diag(np.r_[
                        0,1,0, # Conductivity 
                        0,1,0, # chargeability
                        0,1,0, # time constant
                        0,1,0, # exponent C
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]

            m_fix = np.r_[-np.log(res_sea),0, -np.log(res_base), # Conductivity
                        eta_sea, 0, eta_base, # chargeability
                        np.log(tau_sea), 0, np.log(tau_base),# time constant
                        c_sea, 0, c_base,# exponent C
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP= Prj_m.shape[0]
            self.nM= Prj_m.shape[1]
            self.nM_r, self.nM_m  = 1,1
            self.nM_t, self.nM_c, self.nM_d = 1,1 , 0
            self.res_sea, self.con_sea = res_sea, 1/res_sea
            self.eta_sea, self.tau_sea, self.c_sea = eta_sea, tau_sea, c_sea

        if self.ip_model == 'DDR':
            pattern_eta = list(np.r_[0,1,0])
            diag_etas = np.array(pattern_eta*self.ntau).flatten()
            Prj_m_diag = np.diag(np.r_[
                        0,1,0,  # resistivity
                        diag_etas, # etas
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]
            m_fix_etas_pattern = list(np.r_[eta_sea, 0,eta_base])
            m_fix_etas = np.array(m_fix_etas_pattern*self.ntau).flatten()
            m_fix = np.r_[np.log(res_sea),0, np.log(res_base), # resistivity
                        m_fix_etas, # etas
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP, self.nM = Prj_m.shape
            self.nM_r, self.nM_m= 1, self.ntau
            self.nM_t, self.nM_c, self.nM_d = 0, 0, 0
            self.res_sea, self.con_sea, self.eta_sea, = res_sea, 1/res_sea, eta_sea
 
        if self.ip_model == 'DDC':
            pattern_eta = list(np.r_[0,1,0])
            diag_etas = np.array(pattern_eta*self.ntau).flatten()
            Prj_m_diag = np.diag(np.r_[
                        0, 1, 0,  # Conductivity
                        diag_etas, # etas
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]
            m_fix_etas_pattern = list(np.r_[eta_sea,0,eta_base])
            m_fix_etas = np.array(m_fix_etas_pattern*self.ntau).flatten()
            m_fix = np.r_[-np.log(res_sea),0, -np.log(res_base), # resistivity
                        m_fix_etas, # etas
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP, self.nM = Prj_m.shape
            self.nM_r, self.nM_m= 1, self.ntau
            self.nM_t, self.nM_c, self.nM_d = 0, 0, 0
            self.res_sea, self.con_sea, self.eta_sea, = res_sea, 1/res_sea, eta_sea

    def deepsea_one_IP_layer(self, res_sea,
         eta_sea=0,  tau_sea=1e-3, c_sea=0.4,
         eta_base=0, tau_base=1e-2, c_base=0.4
         ):
        """
        Sea water as fixed layer
        Target layer as IP layer with # self.nlayer-1
        Non IP bottom layer 
        m[         0:     nlayer] : resistivity of the layer
        m[  nlayer  : 2*nlayer-1] : resistivity of bottom layer in log
        m[2*nlayer-1: 3*nlayer-3] : chargeability of the layer
        m[3*nlayer-3: 4*nlayer-5] : time constant of the layer in log
        m[4*nlayer-5: 5*nlayer-7] : exponent C of the layer
        m[5*nlayer-7:] : Thickness of the layer in log
        """
        nlayer_fix =1
        nlayer_sum = self.nlayer + nlayer_fix
        # resistivity
        if self.ip_model == 'pelton':
            Prj_m_diag = np.diag(np.r_[
                        0,np.ones(self.nlayer  ),  # resistivity 
                        0,np.ones(self.nlayer-1),0, # chargeability
                        0,np.ones(self.nlayer-1),0, # time constant
                        0,np.ones(self.nlayer-1),0, # exponent C
                        np.ones(self.nlayer-1) # Thickness
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]

            m_fix = np.r_[np.log(res_sea),np.zeros(self.nlayer), # resistivity
                        eta_sea, np.zeros(self.nlayer-1), eta_base, # chargeability
                        np.log(tau_sea), np.zeros(self.nlayer-1), np.log(tau_base),# time constant
                        c_sea, np.zeros(self.nlayer-1), c_base,# exponent C
                        np.zeros(self.nlayer-1)
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP= Prj_m.shape[0]
            self.nM= Prj_m.shape[1]
            self.nM_r, self.nM_m  = self.nlayer, self.nlayer-1 
            self.nM_t, self.nM_c, self.nM_d = self.nlayer-1, self.nlayer-1, self.nlayer-1
            self.res_sea, self.con_sea = res_sea, 1/res_sea
            self.eta_sea, self.tau_sea, self.c_sea = eta_sea, tau_sea, c_sea

        if self.ip_model == 'Cole':
            Prj_m_diag = np.diag(np.r_[
                        0,np.ones(self.nlayer  ),  # Conductivity 
                        0,np.ones(self.nlayer-1),0, # chargeability
                        0,np.ones(self.nlayer-1),0, # time constant
                        0,np.ones(self.nlayer-1),0, # exponent C
                        np.ones(self.nlayer-1) # Thickness
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]

            m_fix = np.r_[np.log(1.0/res_sea),np.zeros(self.nlayer), # resistivity
                        eta_sea, np.zeros(self.nlayer-1), eta_base, # chargeability
                        np.log(tau_sea), np.zeros(self.nlayer-1), np.log(tau_base),# time constant
                        c_sea, np.zeros(self.nlayer-1), c_base,# exponent C
                        np.zeros(self.nlayer-1)
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP= Prj_m.shape[0]
            self.nM= Prj_m.shape[1]
            self.nM_r, self.nM_m  = self.nlayer, self.nlayer-1 
            self.nM_t, self.nM_c, self.nM_d = self.nlayer-1, self.nlayer-1, self.nlayer-1
            self.res_sea, self.con_sea = res_sea, 1/res_sea
            self.eta_sea, self.tau_sea, self.c_sea = eta_sea, tau_sea, c_sea

        if self.ip_model == 'DDR':
            pattern_eta = list(np.r_[0,np.ones(self.nlayer-1),0])
            diag_etas = np.array(pattern_eta*self.ntau).flatten()
            Prj_m_diag = np.diag(np.r_[
                        0,np.ones(self.nlayer  ),  # resistivity
                        diag_etas, # etas
                        np.ones(self.nlayer-1) # Thickness
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]
            m_fix_etas_pattern = list(np.r_[eta_sea, np.zeros(self.nlayer-1),eta_base])
            m_fix_etas = np.array(m_fix_etas_pattern*self.ntau).flatten()
            m_fix = np.r_[np.log(res_sea),np.zeros(self.nlayer), # resistivity
                        m_fix_etas, # etas
                        np.zeros(self.nlayer-1) # Thickness
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP, self.nM = Prj_m.shape
            self.nM_r, self.nM_m= self.nlayer, self.ntau*(self.nlayer-1)
            self.nM_t, self.nM_c, self.nM_d = 0, 0, self.nlayer-1
            self.res_sea, self.con_sea, self.eta_sea, = res_sea, 1/res_sea, eta_sea
 
        if self.ip_model == 'DDC':
            pattern_eta = list(np.r_[0,np.ones(self.nlayer-1),0])
            diag_etas = np.array(pattern_eta*self.ntau).flatten()
            Prj_m_diag = np.diag(np.r_[
                        0,np.ones(self.nlayer  ),  # resistivity
                        diag_etas, # etas
                        np.ones(self.nlayer-1) # Thickness
                        ])
            non_zero_columns = ~np.all(Prj_m_diag == 0, axis=0)
            Prj_m = Prj_m_diag[:, non_zero_columns]
            m_fix_etas_pattern = list(np.r_[eta_sea, np.zeros(self.nlayer-1),eta_base])
            m_fix_etas = np.array(m_fix_etas_pattern*self.ntau).flatten()
            m_fix = np.r_[np.log(1./res_sea),np.zeros(self.nlayer), # resistivity
                        m_fix_etas, # etas
                        np.zeros(self.nlayer-1) # Thickness
                        ]
            self.nlayer_fix = nlayer_fix
            self.Prj_m = Prj_m
            self.m_fix = m_fix
            self.nP, self.nM = Prj_m.shape
            self.nM_r, self.nM_m= self.nlayer, self.ntau*(self.nlayer-1)
            self.nM_t, self.nM_c, self.nM_d = 0, 0, self.nlayer-1
            self.res_sea, self.con_sea, self.eta_sea, = res_sea, 1/res_sea, eta_sea

    def fix_sea_one_tau_c(self,
             res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([ # res chg
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m_b =np.block([ # tau c
            [0], # sea water
            [np.ones(nlayer).reshape(-1,1)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, nlayer+2))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_b, np.zeros(nlayer_sum).reshape(-1,1)], # Time constant
        [np.zeros((nlayer_sum,2*nlayer)), np.zeros(nlayer_sum).reshape(-1,1), Prj_m_b], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer) # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = 1
        self.nM_c = 1
        assert self.nP == 4*(nlayer_sum)
        assert self.nM == 2*nlayer + 2
        return Prj_m, m_fix

    def fix_sea(self, res_sea, chg_sea, tau_sea, c_sea):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=1
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, 3*nlayer))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2*nlayer))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_A, np.zeros((nlayer_sum, nlayer))], # Time constant
        [np.zeros((nlayer_sum,3*nlayer)), Prj_m_A], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), # Resistivity
        chg_sea, np.zeros(nlayer), # Chargeability
        np.log(tau_sea),np.zeros(nlayer), # Time constant
        c_sea,np.zeros(nlayer)# Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = nlayer
        self.nM_c = nlayer
        assert self.nP == 4*(nlayer+nlayer_fix)
        assert self.nM == 4*nlayer
        return Prj_m, m_fix

    def noIP(self):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=0
        nlayer_sum = nlayer+nlayer_fix
        taumin = self.taumin
        cmin = self.cmin
        Prj_m=np.block([
        [np.eye(nlayer)], # Resistivity
        [np.zeros((nlayer,nlayer))], # Chargeability
        [np.zeros((nlayer,nlayer))], # Time constant
        [np.zeros((nlayer,nlayer))], # Exponent C
        ])
        m_fix = np.r_[ 
        np.zeros(nlayer), # Resistivity
        np.zeros(nlayer), # Chargeability
        np.log(taumin)*np.ones(nlayer), # Time constant
        cmin*np.ones(nlayer)# Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        self.nM_r = nlayer
        self.nM_m = nlayer
        self.nM_t = nlayer
        self.nM_c = nlayer
        assert self.nP == 4*(nlayer_sum)
        assert self.nM == nlayer
        return Prj_m, m_fix

    def pelton_et_al(self, inp, p_dict):
        """ 
        https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py
        """
        # Compute complex resistivity from Pelton et al.
        iotc = np.outer(2j * np.pi * p_dict['freq'], inp['tau']) ** inp['c']
        rhoH = inp['rho_0'] * (1 - inp['m'] * (1 - 1 / (1 + iotc)))
        rhoV = rhoH * p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = 1 / rhoH + 1j * p_dict['etaH'].imag
        etaV = 1 / rhoV + 1j * p_dict['etaV'].imag
        return etaH, etaV

    def cole_cole(self, inp, p_dict):
        """ 
        Slightly modified from the original code in empymod
        (Original) cond_8, cond_0
        (Modified) cond_8, m 
        https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py
        """
        # Compute complex conductivity from Cole-Cole
        iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
        condH = inp['cond_8']*(1.0-  (inp['m'])/(1+iotc))
        condV = condH/p_dict['aniso']**2

        # Add electric permittivity contribution
        etaH = condH + 1j*p_dict['etaH'].imag
        etaV = condV + 1j*p_dict['etaV'].imag

        return etaH, etaV

    def debye_sum_ser(self, inp, p_dict):
        assert self.taus is not None, "taus is not defined"
        assert self.ntau is not None, "ntau is not defined"

        rho0 = inp['rho_0']
        rho0 = rho0.reshape(1,-1) # Shape: [1, nlayer]
        
        etas_list = []

        for i in  range(self.ntau):
            etas_list.append(inp[f'm_{i}']) 
        etas = np.array(etas_list) # Shape: [ntau, nlayer]
        # Reshape etas to [1, nlayer, ntau]
        etas = etas.T[None, :, :]  # Transpose to [nlayer, ntau] → add batch dim → [1, nlayer, ntau]
        assert self.ntau == etas.shape[2], "length of etas must be ntau"

        # Angular frequency, reshaped to [nfreq, 1, 1]
        omega = 2.0 * np.pi * p_dict['freq']
        omega = omega.reshape(-1, 1, 1)

        taus = self.taus.reshape(1, 1, -1) # Shape: [1, 1, ntau]
        iwt = 1.0j * omega * taus # Shape: [nfreq, 1, ntau]
        term = etas / (1.0 + iwt)  # shape: [nfreq, nlayer, ntau]

        # Compute effective resistivity
        # etas.sum(axis=2) has shape [1, nlayer], broadcasted to [nfreq, nlayer]
        rhoH = rho0  * (1.0 - etas.sum(axis=2) + term.sum(axis=2))
        # print(rhoH.shape)
        # print(rhoH)

        rhoV = rhoH * p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = 1 / rhoH + 1j * p_dict['etaH'].imag
        etaV = 1 / rhoV + 1j * p_dict['etaV'].imag
        return etaH, etaV

    def debye_sum_par(self, inp, p_dict):
        assert self.taus is not None, "taus is not defined"
        assert self.ntau is not None, "ntau is not defined"

        con8 = inp['cond_8']
        con8 = con8.reshape(1,-1) # Shape: [1, nlayer]
        
        etas_list = []
        for i in  range(self.ntau):
            etas_list.append(inp[f'm_{i}']) 
        etas = np.array(etas_list) # Shape: [ntau, nlayer]
        # Reshape etas to [1, nlayer, ntau]
        etas = etas.T[None, :, :]  # Transpose to [nlayer, ntau] → add batch dim → [1, nlayer, ntau]
        assert self.ntau == etas.shape[2], "length of etas must be ntau"

        # Angular frequency, reshaped to [nfreq, 1, 1]
        omega = 2.0 * np.pi * p_dict['freq']
        omega = omega.reshape(-1, 1, 1)

        taus = self.taus.reshape(1, 1, -1) # Shape: [1, 1, ntau]
        iwt = 1.0j * omega * taus # Shape: [nfreq, 1, ntau]
        term = etas / (1.0 + iwt)  # shape: [nfreq, nlayer, ntau]

        # Compute effective resistivity
        # etas.sum(axis=2) has shape [1, nlayer], broadcasted to [nfreq, nlayer]
        condH= con8 * (1.0 - term.sum(axis=2))
        condV = condH/p_dict['aniso']**2

        # Add electric permittivity contribution
        etaH = condH + 1j*p_dict['etaH'].imag
        etaV = condV + 1j*p_dict['etaV'].imag
        return etaH, etaV

    def get_ip_model(self, mvec):
        nlayer= self.nlayer
        nlayer_fix = self.nlayer_fix
        nlayer_sum = nlayer + nlayer_fix
        param = self.Prj_m @ mvec + self.m_fix
        if self.ip_model == 'pelton':
            res = np.exp(param[            :   nlayer_sum])
            m   =        param[  nlayer_sum: 2*nlayer_sum]
            tau = np.exp(param[2*nlayer_sum: 3*nlayer_sum])
            c   =        param[3*nlayer_sum: 4*nlayer_sum]
            res_ip = {'res': res, 'rho_0': res, 'm': m,
                    'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}
            if self.m_depth:
                thick = np.exp(param[4*nlayer_sum:])
                depth = self.tx_height + np.r_[0,thick.cumsum()]
                self.model_base['depth'] = depth
            return res_ip

        if self.ip_model == 'Cole':
            con8= np.exp(param[            :   nlayer_sum])
            m   =        param[  nlayer_sum: 2*nlayer_sum]
            tau = np.exp(param[2*nlayer_sum: 3*nlayer_sum])
            c   =        param[3*nlayer_sum: 4*nlayer_sum]
            res_ip = {'res': 1/con8/(1 - m), 'cond_8': con8, 'm': m, 
                      'tau': tau, 'c': c, 'func_eta': self.cole_cole}
            if self.m_depth:
                thick = np.exp(param[4*nlayer_sum:])
                depth = self.tx_height + np.r_[0,thick.cumsum()]
                self.model_base['depth'] = depth
            return res_ip

        elif self.ip_model == 'DDR':
            res = np.exp(param[            :   nlayer_sum])
            res_ip = {'res': res, 'rho_0': res, 'func_eta': self.debye_sum_ser}
            m_list = []
            for i in range(self.ntau):
                etai = param[(i+1)*nlayer_sum:(i+2)*nlayer_sum] 
                res_ip[f"m_{i}"] = etai
                m_list.append(etai)
            m = np.array(m_list)
            m_sum = np.sum(m, axis=0)
            res_ip['m'] = m_sum
            if self.m_depth:
                thick = np.exp(param[-(self.nlayer-1):])
                depth = self.tx_height + np.r_[0,thick.cumsum()]
                self.model_base['depth'] = depth
            return res_ip

        elif self.ip_model == 'DDC':
            con8 = np.exp(param[            :   nlayer_sum])
            res_ip = {'cond_8': con8, 'func_eta': self.debye_sum_par}
            m_list = []
            for i in range(self.ntau):
                etai = param[(i+1)*nlayer_sum:(i+2)*nlayer_sum] 
                res_ip[f"m_{i}"] = etai
                m_list.append(etai)
            m = np.array(m_list)
            m_sum = np.sum(m, axis=0)
            res_ip['m'] = m_sum
            res_ip['res'] = 1/con8/(1 - m_sum)
            if self.m_depth:
                thick = np.exp(param[-(self.nlayer-1):])
                depth = self.tx_height + np.r_[0,thick.cumsum()]
                self.model_base['depth'] = depth
            return res_ip

        else:
            raise ValueError(f"Invalid ip model '{self.ip_model}'. Choose from {self.AVAILABLE_MODELS}")

    def dpred(self,m):
        return self.predicted_data(m)

    def predicted_data(self, model_vector):
        cut_off = self.cut_off
        filt_curr = self.filt_curr
        window_mat = self.window_mat
        data = empymod.bipole(res=self.get_ip_model(model_vector), **self.model_base)
        if data.ndim == 3:
            # Sum over transmitter and receiver dimensions (axis 1 and axis 2)
            data=np.sum(data, axis=(1, 2))
        elif data.ndim == 2:
            # Sum over the transmitter dimension (axis 1)
            if self.recw is None:
                data= np.sum(data, axis=1)
            else:
                recw = self.recw
                data =(recw @ data.T).squeeze()

        self.nD = len(data)
        if cut_off is not None:
            times = self.model_base['freqtime']
            smp_freq = 1/(times[1]-times[0])
            data_LPF = self.apply_lowpass_filter(
                       data=data,cut_off=cut_off,smp_freq=smp_freq
                       )
            data = data_LPF
        if filt_curr is not None:
            data_curr = signal.convolve(data_LPF, filt_curr)[:len(data)]
            data = data_curr
        if window_mat is not None:
            data_window = window_mat @ data_curr
            self.nD = len(data_window)
            data = data_window
        return data
   
    def apply_lowpass_filter(self, data, cut_off,smp_freq, order=1):
        nyquist = 0.5 * smp_freq
        normal_cutoff = cut_off / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def projection_halfspace(self, a, x, b):
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x
    
    def project_convex_set(self, m):
        return self.clip_model(m)

    def proj_c(self,mvec):
        "Project model vector to convex set defined by bound information"
        nlayer = self.nlayer
        a = np.r_[1]
        print(mvec)
        for j in range(nlayer):
            r_prj = mvec[j]
            m_prj = mvec[j+   nlayer]
            t_prj = mvec[j+ 2*nlayer]
            c_prj = mvec[j+ 3*nlayer]
            r_prj = float(self.projection_halfspace( a, r_prj,  np.log(self.resmax)))
            r_prj = float(self.projection_halfspace(-a, r_prj, -np.log(self.resmin)))
            m_prj = float(self.projection_halfspace( a, m_prj,  self.chgmax))
            m_prj = float(self.projection_halfspace(-a, m_prj, -self.chgmin))
            t_prj = float(self.projection_halfspace( a, t_prj,  np.log(self.taumax)))
            t_prj = float(self.projection_halfspace(-a, t_prj, -np.log(self.taumin)))
            c_prj = float(self.projection_halfspace( a, c_prj,  self.cmax))
            c_prj = float(self.projection_halfspace(-a, c_prj, -self.cmin))
            mvec[j         ] = r_prj
            mvec[j+  nlayer] = m_prj
            mvec[j+2*nlayer] = t_prj
            mvec[j+3*nlayer] = c_prj
        return mvec
  
    def clip_model(self, mvec):
        mvec_tmp = mvec.copy()
        # nlayer = self.nlayer
        index_r = self.nM_r
        index_m = self.nM_m +index_r
        index_t = self.nM_t + index_m
        index_c = self.nM_c + index_t
        mvec_tmp[        : index_r]=np.clip(
            mvec[        : index_r], np.log(self.resmin), np.log(self.resmax)
            )
        if self.nM_m > 0:
            mvec_tmp[ index_r: index_m]=np.clip(
                mvec[ index_r: index_m], self.chgmin, self.chgmax
                )
        if self.nM_t > 0:
            mvec_tmp[ index_m: index_t]=np.clip(
                mvec[ index_m: index_t], np.log(self.taumin), np.log(self.taumax)
                )
        if self.nM_c > 0: 
            mvec_tmp[ index_t: index_c]=np.clip(
                mvec[ index_t: index_c], self.cmin, self.cmax
                )
        return mvec_tmp
    
    def J(self, model_vector):
        return self.Japprox(model_vector)

    def Japprox(self, model_vector, perturbation=0.1, min_perturbation=1e-3):
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])
#        delta_m = perturbation  # np.max([perturbation*m.mean(), min_perturbation])
        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            pos = self.predicted_data(mpos)
            neg = self.predicted_data(mneg)
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T
    
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

    def plot_model(self, model, depth_min=-1e3,depth_max=1e3, ax=None, **kwargs):
        """
        Plot a single model (e.g., resistivity, chargeability) with depth.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Default plotting parameters
        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "label": "model",
        }
        default_kwargs.update(kwargs)
        if self.nlayer + self.nlayer_fix == 1:
            depth = np.r_[depth_min, depth_max ]
        # Prepare depth and model data for plotting
        else:
            depth = np.r_[depth_min + self.model_base["depth"][0], 
                        self.model_base["depth"],
                        depth_max + self.model_base["depth"][-1] ]
        depth_plot = np.vstack([depth, depth]).flatten(order="F")[1:-1]
 #       depth_plot = np.hstack([depth_plot, depth_plot[-1] * 1.5])  # Extend depth for plot
        model_plot = np.vstack([model, model]).flatten(order="F")

        # Plot model with depth
        ax.plot(model_plot, depth_plot, **default_kwargs)
        return ax
    
    def plot_IP_par(self, mvec, ax=None, label=None, rm=False, **kwargs):
        """
        Plot all IP parameters (resistivity, chargeability, time constant, exponent c).
        """
        if rm:
            if ax is None:
                fig, ax = plt.subplots(1, 2, figsize=(12, 8))  # Create 2x2 grid of subplots
            else:
                ax = np.array(ax)  # Convert ax to a NumPy array if it's not already
                ax = ax.flatten()  # Ensure ax is a flat array
            # Convert model vector to parameters
            model = self.get_ip_model(mvec)

            # Plot each model parameter
            self.plot_model(model["res"], ax=ax[0], label=label, **kwargs)
            ax[0].set_title("Resistivity (ohm-m)")

            self.plot_model(model["m"], ax=ax[1], label=label, **kwargs)
            ax[1].set_title("Chargeability")
            return ax
        
        else:
            if ax is None:
                fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # Create 2x2 grid of subplots
            else:
                ax = np.array(ax)  # Convert ax to a NumPy array if it's not already
                ax = ax.flatten()  # Ensure ax is a flat array


            # Convert model vector to parameters
            model = self.get_ip_model(mvec)

            # Plot each model parameter
            self.plot_model(model["res"], ax=ax[0], label=label, **kwargs)
            ax[0].set_title("Resistivity (ohm-m)")

            self.plot_model(model["m"], ax=ax[1], label=label, **kwargs)
            ax[1].set_title("Chargeability")

            self.plot_model(model["tau"], ax=ax[2], label=label, **kwargs)
            ax[2].set_title("Time Constant (sec)")

            self.plot_model(model["c"], ax=ax[3], label=label, **kwargs)
            ax[3].set_title("Exponent C")
            return ax

class Pelton_res_f(): 
    def __init__(self, freq=None, con=False,
                 reslim= np.r_[1e-2,1e5],
                 chglim= np.r_[1e-3, 0.9],
                taulim= np.r_[1e-4, 1e0],
                clim= np.r_[0.20, 0.9]
                 ):
        self.freq = freq
        self.reslim = np.log(reslim)
        self.chglim = chglim
        self.taulim = np.log(taulim)
        self.clim =  clim
        self.con = con
    
    def f(self,p):
        """
        Pelton in resistivity form in frequency domain.
          `p` (`torch.Tensor`): Model parameters
            - `p[0]` : log(res0)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c
        ```math
        \rho(\omega) = \rho_0 \left[ 1 - \eta \left(1 - \frac{1}{1+(i\omega\tau)^c} \right) \right]
        = \rho_0 \left[ \frac{\tau^{-c} + (1-\eta)(i\omega)^c}{\tau^{-c} + (i\omega)^c} \right]
        ```
        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwc = (1j * 2. * np.pi * self.freq  ) ** p[3] 
        tc = np.exp(-p[2]*p[3])
        if self.con:
            return np.exp(-p[0])/(tc +(1.0-p[1])*iwc)*(tc+iwc)
        else:
            return np.exp( p[0])*(tc +(1.0-p[1])*iwc)/(tc+iwc)

    def f_grad(self,p):
        """
        Gradient of the Pelton resistivity form in frequency domain.
        Parameters
        ----------
            Model parameters:
                - p[0] : log(res0)
                - p[1] : eta
                - p[2] : log(tau)
                - p[3] : c
        -----
            \\frac{\\partial \\rho(\\omega)}{\\partial \\rho_0}
            = 1 - \\eta \\left(1 - \\frac{1}{1 + (i \\omega \\tau)^c}\\right)
            \\frac{\\partial \\rho(\\omega)}{\\partial \\eta}
            = -\\rho_0 \\left(1 - \\frac{1}{1 + (i \\omega \\tau)^c}\\right)
            \\frac{\\partial \\rho(\\omega)}{\\partial \\tau}
            = \\rho_0 \\eta \\left[ \\frac{-c (i \\omega)^c \\tau^{c-1}}{(1 + (i \\omega \\tau)^c)^2} \\right]
            \\frac{\\partial \\rho(\\omega)}{\\partial c}
            = \\rho_0 \\eta \\left[ \\frac{-(i \\omega \\tau)^c \\log(i \\omega \\tau)}{(1 + (i \\omega \\tau)^c)^2} \\right]

        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwt = 1.j * 2. * np.pi * self.freq * np.exp(p[2])
        iwc = (1.j * 2. * np.pi*  self.freq) ** p[3]
        iwtc = iwt ** p[3]

        # Initialize gradient as a complex tensor
        grad = np.zeros((len(self.freq), len(p)), dtype=np.cfloat)

        # Derivative with respect to res0
        grad[:,0] =  np.exp(p[0]) * (1 - p[1] * (1 - 1. / (1. + iwtc)))
        grad[:,1] = -np.exp(p[0]) * (1 - 1. / (1. + iwtc))

        # Derivatives with respect to tau and c
        if p[1] != 0:
            grad[self.freq!=0,2] = np.exp(p[0]) * p[1] *np.exp(p[2])* (
                -p[3]*iwc[self.freq!=0] * np.exp(p[2]) ** (p[3] - 1)
                 / (1. + iwtc[self.freq!=0]) ** 2)
            grad[self.freq!=0,3] = np.exp(p[0]) * p[1] * (
                -iwtc[self.freq!=0] * np.log(iwt[self.freq!=0])
                  / (1. + iwtc[self.freq!=0]) ** 2)
        if self.con:
            f= self.f(p) # Note f return conductivity
            f= f.reshape(-1,1)  # reshape
            grad *= -f**2 # C' = (1/Z)' = -Z'/Z**2 = -Z'*C**2 
        return grad  

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec       
        mvec_tmp[0] = np.clip(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1] = np.clip(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = np.clip(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3]   = np.clip(mvec[3]  ,   self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton_con_f(): 
    def __init__(self, freq=None, 
                 conlim= np.r_[1e-5,1e2],
                 chglim= np.r_[1e-3, 0.9],
                taulim= np.r_[1e-4, 1e0],
                clim= np.r_[0.20, 0.9]
                 ):
        self.freq = freq
        self.conlim = np.log(conlim)
        self.chglim = chglim
        self.taulim = np.log(taulim)
        self.clim =  clim
    
    def f(self,p):
        """
        Pelton model in conductivity form in Frequency domain
        - `p` (`torch.Tensor`): Model parameters
            - `p[0]` : log(con8)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c
        ```math
        \sigma_\infty  \left[ 1- \dfrac{\eta}{1+(1-\eta)(i\omega \tau)^c} \right]
        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwc = (1j * 2. * np.pi * self.freq  ) ** p[3] 
        tc = np.exp(-p[2]*p[3])
        f = np.exp(p[0])*(1.0-p[1])*(tc+iwc)/(tc +(1.0-p[1])*iwc)
        return f

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec       
        mvec_tmp[0] = np.clip(mvec[0], self.conlim.min(), self.conlim.max())
        mvec_tmp[1] = np.clip(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = np.clip(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = np.clip(mvec[3],   self.clim.min(), self.clim.max())
        return mvec_tmp

class Cole_Cole_res_f(): 
    def __init__(self, freq=None, 
                 reslim= np.r_[1e-2,1e5],
                 chglim= np.r_[1e-3, 0.9],
                taulim= np.r_[1e-4, 1e0],
                clim= np.r_[0.20, 0.9]
                 ):
        self.freq = freq
        self.reslim = np.log(reslim)
        self.chglim = chglim
        self.taulim = np.log(taulim)
        self.clim =  clim
    
    def f(self,p):
        """
        Cole-Cole model in resistivity form in frequency domain
        - `p` (`torch.Tensor`): Model parameters

            - `p[0]` : log(res0)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c
        ```math
        \rho_0(1-\eta) \left[1+ \dfrac{\eta}{1-\eta+(i\omega \tau)^c}\right]
        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwc = (1j * 2. * np.pi * self.freq  ) ** p[3] 
        tc = np.exp(-p[2]*p[3])
        return np.exp(p[0])*(1.0-p[1])*(tc+iwc)/((1.0-p[1])*tc +iwc)

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec       
        mvec_tmp[0] = np.clip(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1] = np.clip(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = np.clip(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = np.clip(mvec[3],   self.clim.min(), self.clim.max())
        return mvec_tmp

class Cole_Cole_con_f(): 
    def __init__(self, freq=None, res=False,
                 conlim= np.r_[1e-5,1e2],
                 chglim= np.r_[1e-3, 0.9],
                taulim= np.r_[1e-4, 1e0],
                clim= np.r_[0.20, 0.9]
                 ):
        self.freq = freq
        self.conlim = np.log(conlim)
        self.chglim = chglim
        self.taulim = np.log(taulim)
        self.clim =  clim
        self.res = res
    
    def f(self,p):
        """
        Cole-Cole model in conductivity form in frequency domain
        - `p` (`torch.Tensor`): Model parameters
            - `p[0]` : log(con8)
            - `p[1]` : eta
            - `p[2]` : log(tau)
            - `p[3]` : c
        ```math
        \sigma_\infty \left(1- \dfrac{\eta}{1+(i\omega \tau)^c}\right)
        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwc = (1j * 2. * np.pi * self.freq  ) ** p[3] 
        tc = np.exp(-p[2]*p[3])
        if self.res:
            return np.exp(-p[0])/((1.0-p[1])*tc+iwc)*(tc+iwc)
        else:
            return np.exp( p[0])*((1.0-p[1])*tc+iwc)/(tc+iwc)

    def f_grad(self,p):
        """
        Compute the gradient of the Cole-Cole model with respect to its parameters.
        Parameters
        ----------
            Model parameters:
                - p[0] : log(con8)
                - p[1] : eta
                - p[2] : log(tau)
                - p[3] : c
        -----
            \\frac{\\partial \\sigma(\\omega)}{\\partial \\sigma_\\infty}
            = \\left(1 - \\frac{\\eta}{1 + (i \\omega \\tau)^c} \\right)
            \\frac{\\partial \\sigma(\\omega)}{\\partial \\eta}
            = \\frac{-\\sigma_\\infty}{1 + (i \\omega \\tau)^c}
            \\frac{\\partial \\sigma(\\omega)}{\\partial \\tau}
            = \\sigma_\\infty \\eta \\left[ \\frac{c (i \\omega)^c \\tau^{c-1}}{\\left(1 + (i \\omega \\tau)^c\\right)^2} \\right]
            \\frac{\\partial \\sigma(\\omega)}{\\partial c}
            = \\sigma_\\infty \\eta \\left[ \\frac{(i \\omega \\tau)^c \\log(i \\omega \\tau)}{\\left(1 + (i \\omega \\tau)^c\\right)^2} \\right]
        """
        assert len(p) == 4, "Number of parameters must be 4"
        iwt = 1.j * 2. * np.pi * self.freq * np.exp(p[2])
        iwc = (1.j * 2. * np.pi*  self.freq) ** p[3]
        iwtc = iwt ** p[3]

        # Initialize gradient as a complex tensor
        grad = np.zeros((len(self.freq), len(p)), dtype=np.cfloat)

        # Derivative with respect to res0
        grad[:,0] =  np.exp(p[0]) * (1 - p[1] / (1. + iwtc))
        grad[:,1] = -np.exp(p[0])  / (1. + iwtc)

        # Derivatives with respect to tau and c
        if p[1] != 0:
            grad[self.freq!=0,2] = np.exp(p[0]) * p[1] *np.exp(p[2])* (
                p[3]*iwc[self.freq!=0] * np.exp(p[2]) ** (p[3] - 1)
                 / (1. + iwtc[self.freq!=0]) ** 2)
            grad[self.freq!=0,3] = np.exp(p[0]) * p[1] * (
                iwtc[self.freq!=0] * np.log(iwt[self.freq!=0])
                  / (1. + iwtc[self.freq!=0]) ** 2)
        if self.res:
            f= self.f(p) # Note f return resistivity
            f= f.reshape(-1,1)  # reshape
            grad *= -f**2 # Z' = (1/C)' = -C'/C**2 = -C'*Z**2 
        return grad 

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec       
        mvec_tmp[0] = np.clip(mvec[0], self.conlim.min(), self.conlim.max())
        mvec_tmp[1] = np.clip(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = np.clip(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = np.clip(mvec[3] ,  self.clim.min(), self.clim.max())
        return mvec_tmp

class Debye_res_t():
    def __init__(self, times=None, 
            reslim= np.r_[1e-2,1e5],
            chglim= np.r_[1e-3,0.9],
            taulim= np.r_[1e-4,1e0],
            ):
        self.times = times
        self.reslim = np.log(reslim)
        self.chglim = chglim
        self.taulim = np.log(taulim)

    def t(self,p):
        """
        Debye resistivity model.
        p[0] : log(res0)
        p[1] : eta
        p[2] : log(tau)
        """
        t = np.zeros_like(self.times)
        ind_0 = (self.times == 0)
        t[ind_0] = (1.0-p[1]) 
        t[~ind_0] = p[1]/np.exp(p[2])* np.exp(-self.times[~ind_0] /np.exp(p[2]))
        return np.exp(p[0])*t
    
    def t_grad(self,p):
        assert len(p) == 3, "Number of parameters must be 3"
        # Initialize gradient as a complex tensor
        grad = np.zeros((len(self.times), len(p)))

        # Derivative with respect to res0
        ind_0 = (self.times == 0)

        grad[ind_0,0] =  np.exp(p[0]) * (1 - p[1]) 
        grad[ind_0,1] = -np.exp(p[0])

        # Derivatives with respect to tau and c
        if p[1] != 0:
            grad[~ind_0,0] = np.exp(p[0])*p[1]/np.exp(p[2])*np.exp(
                -self.times[~ind_0] /np.exp(p[2])
                )
            grad[~ind_0,1] = np.exp(p[0]) /np.exp(p[2])*np.exp(
                -self.times[~ind_0] /np.exp(p[2])
                )
            grad[~ind_0,2] = np.exp(p[0]) * p[1] * (
                self.times[~ind_0]-np.exp(p[2]))* np.exp(
                -self.times[~ind_0] /np.exp(p[2])
                )  / np.exp(p[2]*2)
        return grad  

class Debye_decmp_res_f:
    def __init__(self,
                 freq=None, taus= None, con=False, 
                 reslim= np.r_[1e-3,1e6],
                 chglim = np.r_[0,0.9] ,
                 ):
        self.freq = freq
        self.taus= taus
        self.con = con
        self.ntau = len(taus)
        self.reslim = np.log(reslim)
        self.chglim = chglim    
        self.proj_a = np.ones(self.ntau)

    def f(self,p):
        """
        Debye Decomposition model in resistivity form.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \rho_0 \left[ 1 -\sum_{j=1}^n \eta_j + \sum_{j=1}^n \dfrac{\eta_j}{1+i\omega\tau_j}\right] 
        """
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        rho0 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        omega = 2.0 * np.pi * self.freq
        omega = omega.reshape(-1, 1) # shape: [nfreq, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        iwt = 1.0j * omega * taus # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt) # shape: [nfreq, ntau]
        if self.con: 
            return 1.0/rho0 / (1.0 -etas.sum(axis=1)+ term.sum(axis=1)) # shape: [nfreq]
        else:
            return rho0 * (1.0 -etas.sum(axis=1)+ term.sum(axis=1))  # shape: [nfreq]

    def f_grad(self,p):
        '''
        Gradient of Debye Decomposition model in resistivity form in frequency domain.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \dfrac{\partial\rho(\omega)}{\partial \rho_0}= 1-\sum_{j=1}^n \eta_j+ \sum_{j=1}^n \dfrac{\eta_j}{1+(i\omega\tau_j)}}
        \dfrac{\partial\rho(\omega)}{\partial \eta_j}= \rho_0 \left[ -1 + \dfrac{1}{1+(i\omega\tau_j)}\right]
        '''
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        # Initialize gradient as a complex tensor
        rho0 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        omega = 2.0 * np.pi * self.freq
        omega = omega.reshape(-1, 1) # shape: [nfreq, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        iwt = 1.0j * omega * taus # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt) # shape: [nfreq, ntau]
        grad_rho = rho0*(1.0 - etas.sum(axis=1) + term.sum(axis=1))# shape: [nfreq]
        grad_etas = -rho0*iwt/ (1.0 + iwt) # shape: [nfreq, ntau]
        grad = np.concatenate((grad_rho.reshape(-1, 1), grad_etas), axis=1) # shape: [nfreq, ntau+1]
        if self.con:
            f= self.f(p) # Note f return conductivity
            f= f.reshape(-1,1)  # reshape
            grad *= -f**2 # C' = (1/Z)' = -Z'/Z**2 = -Z'*C**2 
        return grad 

    def clip_model(self,mvec):
        mvec_tmp = mvec
        mvec_tmp[0] = np.clip(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau],  self.proj_a, self.chglim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau], -self.proj_a, self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        proj_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        return proj_x

class Debye_decmp_res_t:
    def __init__(self,
                 times=None, tstep=None,taus= None, 
                 reslim= np.r_[1e-3,1e6],
                 chglim = np.r_[0,0.9] ,
                 ):
        self.times = times
        self.tstep = tstep
        self.taus= taus
        self.ntau = len(taus)
        self.reslim = np.log(reslim)
        self.chglim = chglim    
        self.proj_a = np.ones(self.ntau)

    def t(self,p):
        """
        Debye Decomposition model in resistivity form in time domain.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \rho(t)=\rho_0 \left[ \left(1 -\sum_{j=1}^n \eta_j \right) \delta(t)+ \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
        """
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"

        rho0 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        times = self.times.reshape(-1, 1) # shape: [ntime, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        ind_0 = (times == 0)
        term = etas / taus*np.exp(-times/taus) # shape: [ntime, ntau]
        term_sum = term.sum(axis=1) # shape: [ntime]
        term_sum[ind_0] = 1.0-etas.sum(axis=1) # shape: [ntime]
        if self.tstep is not None:
            ind_pos = (times > 0)
            term_sum[ind_pos] *= self.tstep
        return rho0 * term_sum # shape: [ntime]

    def t_grad(self,p):
        '''
        Gradient of Debye Decomposition model in resistivity form in time domain.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \dfrac{\partial\rho(t)}{\partial \rho_0}=\left(1 -\sum_{j=1}^n \eta_j \right) \delta(t)+ \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}}
        \dfrac{\partial\rho(t)}{\partial \eta_j}=-\rho_0 \left[\delta(t)+ \dfrac{1}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
        '''
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        rho0 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        times = self.times.reshape(-1, 1) # shape: [ntime, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        ind_0 = (times == 0)
        term = etas / taus*np.exp(-times/taus) # shape: [ntime, ntau]
        term_sum = term.sum(axis=1) # shape: [ntime]
        term_sum[ind_0] = 1.0-etas.sum(axis=1)
        grad_rho = rho0* term_sum # shape: [ntime]
        grad_etas = -rho0 / taus*np.exp(-times/taus) # shape: [ntime, ntau]
        grad_etas[ind_0] = -rho0
        grad = np.concatenate((grad_rho.reshape(-1, 1), grad_etas), axis=1) # shape: [ntime, ntau+1]
        if self.tstep is not None:
            ind_pos = (times > 0)
            grad[ind_pos,:] *= self.tstep
        return grad # shape: [ntime, ntau+1]

    def clip_model(self,mvec):
        mvec_tmp = mvec
        mvec_tmp[0] = np.clip(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau],  self.proj_a, self.chglim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau], -self.proj_a, self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        proj_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        return proj_x

class debye_con_t():
    def __init__(self, times=None, 
                 conlim= np.r_[1e-5,1e2],
                 chglim= np.r_[1e-3, 0.9],
                taulim= np.r_[1e-4, 1e0]
                 ):
        self.times = times
        self.conlim = np.log(conlim)
        self.chglim = chglim
        self.taulim = np.log(taulim)
    
    def t(self,p):
        """
        Debye conductivity model.
        p[0] : log(con8)
        p[1] : eta
        p[2] : log(tau)
        """
        assert len(p) == 3, "Number of parameters must be 4"
        t = np.zeros_like(self.times)
        ind_0 = (self.times == 0)
        t[ind_0] = 1.0
        t[~ind_0] = -p[1]/(1.0-p[1])/np.exp(p[2])*np.exp(-self.times[~ind_0]/((1.0-p[1])*np.exp(p[2])))
        return np.log(p[0])*t

    def t_grad(self,p):
        nfreq = len(self.freq)
        assert len(p) == 4, "Number of parameters must be 4"
        iwt = 1.j * 2. * np.pi * self.freq * np.exp(p[2])
        iwc = (1.j * 2. * np.pi*  self.freq) ** p[3]
        iwtc = iwt ** p[3]

        # Initialize gradient as a complex tensor
        grad = np.zeros((len(self.freq), len(p)), dtype=np.cfloat)
        #NotImplemented

        # # Derivative with respect to res0
        # grad[:,0] =  np.exp(p[0]) * (1 - p[1] * (1 - 1. / (1. + iwtc)))
        # grad[:,1] = -np.exp(p[0]) * (1 - 1. / (1. + iwtc))

        # # Derivatives with respect to tau and c
        # if p[1] != 0:
        #     grad[self.freq!=0,2] = np.exp(p[0]) * p[1] *np.exp(p[2])* (
        #         -p[3]*iwc[self.freq!=0] * np.exp(p[2]) ** (p[3] - 1)
        #          / (1. + iwtc[self.freq!=0]) ** 2)
        #     grad[self.freq!=0,3] = np.exp(p[0]) * p[1] * (
        #         -iwtc[self.freq!=0] * np.log(iwt[self.freq!=0])
        #           / (1. + iwtc[self.freq!=0]) ** 2)
        return grad 

    def clip_model(self,mvec):
        mvec_tmp = mvec
       
        mvec_tmp[0] = np.clip(mvec[0], self.conlim.min(), self.conlim.max())
        mvec_tmp[1] = np.clip(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = np.clip(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3]   = np.clip(mvec[3]  ,   self.clim.min(), self.clim.max())
        return mvec_tmp

class Debye_decmp_con_f:
    def __init__(self,
                 freq=None, taus= None, res=False, 
                 conlim= np.r_[1e-5,1e2],
                 chglim = np.r_[0,0.9] ,
                 ):
        self.freq = freq
        self.taus= taus
        self.res = res
        self.ntau = len(taus)
        self.conlim = np.log(conlim)
        self.chglim = chglim    
        self.proj_a = np.ones(self.ntau)

    def f(self,p):
        """
        Debye Decomposition model in conductivity form in frequency domain.
        p[0]        : log(con8)
        p[1:1+ntaus]: etas 
        \sigma(\omega)=\sigma_\infty\left(1- \sum_{j=1}^n\dfrac{\eta_j}{1+i\omega\tau_j}\right)
        """
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        con8 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        omega = 2.0 * np.pi * self.freq
        omega = omega.reshape(-1, 1) # shape: [nfreq, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        iwt = 1.0j * omega * taus # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt) # shape: [nfreq, ntau]
        if self.res:  # return resistivity
            return 1.0/con8 / (1.0 -term.sum(axis=1)) # shape: [nfreq]
        else: # return conductivity
            return con8 * (1.0 - term.sum(axis=1))  # shape: [nfreq]

    def f_grad(self,p):
        '''
        Gradient of Debye Decomposition model in conductivity form in frequency domain.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \dfrac{\partial\sigma(\omega)}{\partial \sigma_\infty}=1- \sum_{j=1}^n \dfrac{\eta_j}{1+(i\omega\tau_j)}}
        \dfrac{\partial\sigma(\omega)}{\partial \eta_j}=  \dfrac{\sigma_\infty }{1+(i\omega\tau_j)}
        '''
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        # Initialize gradient as a complex tensor
        con8 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        omega = 2.0 * np.pi * self.freq
        omega = omega.reshape(-1, 1) # shape: [nfreq, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        iwt = 1.0j * omega * taus # shape: [nfreq, ntau]
        term = etas / (1.0 + iwt) # shape: [nfreq, ntau]
        grad_con = con8*(1.0 - term.sum(axis=1))# shape: [nfreq]
        grad_etas = -con8/ (1.0 + iwt) # shape: [nfreq, ntau]
        grad= np.concatenate((grad_con.reshape(-1, 1), grad_etas), axis=1) # shape: [nfreq, ntau+1]
        if self.res:
            f= self.f(p) # Note f return resistivity
            f= f.reshape(-1,1)  # reshape
            grad *= -f**2 # Z' = (1/C)' = -C'/C**2 = -C'*Z**2 
        return grad 


    def clip_model(self,mvec):
        mvec_tmp = mvec
        mvec_tmp[0] = np.clip(mvec[0], self.conlim.min(), self.conlim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau],  self.proj_a, self.chglim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau], -self.proj_a, self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        proj_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        return proj_x

class Debye_decmp_con_t:
    def __init__(self,
                 times=None, tstep=None,taus= None, 
                 conlim= np.r_[1e-5,1e2],
                 chglim = np.r_[0,0.9] ,
                 ):
        self.times = times
        self.tstep = tstep
        self.taus= taus
        self.ntau = len(taus)
        self.conlim = np.log(conlim)
        self.chglim = chglim    
        self.proj_a = np.ones(self.ntau)

    def t(self,p):
        """
        Debye Decomposition model in conductivity in time domain.
        p[0]        : log(con8)
        p[1:1+ntaus]: etas 
        \sigma(t)=\sigma_\infty \left[ \delta(t)- \sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}\right]
     """
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        con8 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        times = self.times.reshape(-1, 1) # shape: [ntime, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        ind_0 = (times == 0)
        term = -etas/taus*np.exp(-times/taus) # shape: [ntime, ntau]
        term_sum = term.sum(axis=1) # shape: [ntime]
        term_sum[ind_0] = 1.0 # shape: [ntime]
        if self.tstep is not None:
            ind_pos = (times > 0)
            term_sum[ind_pos] *= self.tstep
        return con8 * term_sum # shape: [ntime]

    def t_grad(self,p):
        '''
        Gradient of Debye Decomposition model in conudctivity in time domain.
        p[0]        : log(res0)
        p[1:1+ntaus]: etas 
        \dfrac{\partial\sigma(t)}{\partial\sigma_\infty}= \delta(t)-\sum_{j=1}^n \dfrac{\eta_j}{\tau_j}e^{\frac{-t}{\tau_j}}}
        \dfrac{\partial\sigma(t)}{\partial\eta_j}=\dfrac{-\sigma_\infty}{\tau_j}e^{\frac{-t}{\tau_j}}        '''
        assert len(p) == 1 + self.ntau, "Number of parameters must match number of taus"
        assert self.res == False, "This function is only for conductivity"
        con8 = np.exp(p[0])
        etas = p[1:1+self.ntau]
        etas= etas.reshape(1, -1) # shape: [1, ntau]
        times = self.times.reshape(-1, 1) # shape: [ntime, 1]
        taus = self.taus.reshape(1, -1) # shape: [1, ntau]
        ind_0 = (times == 0)
        term = -etas / taus*np.exp(-times/taus) # shape: [ntime, ntau]
        term_sum = term.sum(axis=1) # shape: [ntime]
        term_sum[ind_0] = 1.0
        grad_rho = con8 *term_sum # shape: [ntime]
        grad_etas = -con8/taus*np.exp(-times/taus) # shape: [ntime, ntau]
        grad = np.concatenate((grad_rho.reshape(-1, 1), grad_etas), axis=1) # shape: [ntime, ntau+1]
        if self.tstep is not None:
            ind_pos = (times > 0)
            grad[ind_pos,:] *= self.tstep
        return grad # shape: [ntime, ntau+1]

    def clip_model(self,mvec):
        mvec_tmp = mvec
        mvec_tmp[0] = np.clip(mvec[0], self.conlim.min(), self.conlim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau],  self.proj_a, self.chglim.max())
        mvec_tmp[1:1+self.ntau] = self.proj_halfspace(mvec_tmp[1:1+self.ntau], -self.proj_a, self.chglim.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        proj_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        return proj_x

class InducedPolarizationSimulation(BaseSimulation):
    AVAILABLE_MODES = ['tdip_t', 'tdip_f', 'sip_t', 'sip']
    def __init__(self, 
                 ip_model=None,
                 mode="sip",
                 window_mat=None,
                 ):
        self.ip_model = ip_model
        self.mode = mode
        self.window_mat = window_mat    

    def dpred(self,m):
        if self.mode=="sip":
            f = self.ip_model.f(m)
            f_real = f.real
            f_imag = f.imag
            if self.window_mat is None:
                return np.r_[f_real, f_imag] 
            else:
                return np.r_[self.window_mat@f_real, self.window_mat@f_imag]  
    
    def J(self,m):
        if self.mode=="sip":
            f_grad = self.ip_model.f_grad(m)
            f_real_grad = f_grad.real
            f_imag_grad = f_grad.imag
            if self.window_mat is None:
                return np.vstack([f_real_grad, f_imag_grad])
            else:
                return np.vstack([self.window_mat@f_real_grad, self.window_mat@f_imag_grad])

    def J_prd(J):
        """"
        Retursn two arrays
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
    
    def project_convex_set(self,m):
        return self.ip_model.clip_model(m)

# class Optimization(BaseSimulation):  # Inherits from BaseSimulation
class Optimization:  # Inherits from BaseSimulation
    def __init__(self,
                sim, # BaseSimulation,
                dobs=None,
                Wd=None, Ws=None, Wx=None, alphas=1, alphax=1,
                Ws_threshold=1e-6, eig_tol=1e-6
                ):
        self.sim = sim  # Composition: opt_tmp has a simulation
        self.dobs= dobs
        self.Wd = Wd
        self.Ws = Ws
        self.Ws_threshold = Ws_threshold
        self.Wx = Wx
        self.eig_tol = eig_tol
        self.alphas = alphas
        self.alphax = alphax

    def dpred(self, m):
        return self.sim.dpred(m)  # Calls InducedPolarization's dpred()

    def J(self, m):
        return self.sim.J(m)  # Calls InducedPolarization's J()
    
    def project_convex_set(self,m):
        return self.sim.project_convex_set(m)

    def get_Wd(self,ratio=0.10, plateau=0):
        dobs_clone = self.dobs.copy()
        noise_floor = plateau * np.ones_like(dobs_clone)
        std = np.sqrt(noise_floor**2 + (ratio * np.abs(dobs_clone))**2)
        self.Wd =np.diag(1 / std.flatten())
        return self.Wd
    
    def get_Ws(self, smallness):
        # if self.sim.nM is not None:
        #     self.Ws = np.eye(self.sim.nM)
        # else:
        self.smallness= smallness
        self.Ws = np.diag(smallness)

    def update_Ws(self, J):
        if self.Wd is None:
            Sensitivity = np.sqrt(np.sum(J**2, axis=0))
        else:
            Sensitivity = np.sqrt(np.sum((self.Wd@J)**2, axis=0))
        # Sensitivity = self.compute_sensitivity(self.Wd@J)
        Sensitivity /= Sensitivity.max()
        Sensitivity = np.clip(Sensitivity, self.Ws_threshold, 1)
        self.Ws = np.diag(self.smallness*Sensitivity)

    def get_Wx_Debye_sum(self, mvec):
        Wx = np.zeros((len(mvec)-3, len(mvec)))
        Wx[:,2:-1] = -np.diag(np.ones(len(mvec)-3))
        Wx[:,3:] += np.diag(np.ones(len(mvec)-3))
        self.Wx = Wx
        return Wx
    
    def get_Wx(self):
        nlayer = self.sim.nlayer
        depth = self.sim.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        if nlayer == 1:
            nM = self.nM
            nP = self.nP
            Wx = np.zeros((nM,nP))
            print("No smoothness for one layer model")
            self.Wx = Wx
            return Wx
        Wx_block = np.zeros((nlayer-1, nlayer))
        delta_x = np.diff(depth)
        elm1 = 1/delta_x[:-1]
        Wx_block[:,:-1] = -np.diag(elm1)
        Wx_block[:,1:] += np.diag(elm1)
        elm2 = np.sqrt(delta_x)
        Wx_block = Wx_block @ np.diag(elm2)
        # Wx_block = np.zeros((nlayer-1, nlayer))
        # Wx_block[:,1:-1] = np.eye(nlayer-1)
        # Wx_block[:,2:] -= np.eye(nlayer-1)
        Wx=np.block([
        [Wx_block, np.zeros((nlayer-1, nlayer*3))], # Resistivity
        [np.zeros((nlayer-1, nlayer*1)), Wx_block, np.zeros((nlayer-1, nlayer*2))], # Chargeability
        [np.zeros((nlayer-1, nlayer*2)), Wx_block, np.zeros((nlayer-1, nlayer*1))], # Time constant
        [np.zeros((nlayer-1, nlayer*3)), Wx_block], # Exponent C
        ])
        self.Wx = Wx
        return Wx

    def get_Wx_sea_basement(self):
        nlayer = self.sim.nlayer
        nlayer_fix=2
        nlayer_sum = nlayer+nlayer_fix
        nM = self.sim.nM
        nP = self.sim.nP
        Wx = np.zeros((nM,nP))
        if nlayer == 1:
            print("No smoothness for one layer model")
            self.Wx = Wx
            return Wx
        Wx_block = np.zeros((nlayer-1, nlayer_sum))
        Wx_block[:,1:-2] = np.eye(nlayer-1)
        Wx_block[:,2:-1] -= np.eye(nlayer-1)
        Wx=np.block([
        [Wx_block, np.zeros((nlayer-1, nlayer_sum*3))], # Resistivity
        [np.zeros((nlayer-1, nlayer_sum*1)), Wx_block, np.zeros((nlayer-1, nlayer_sum*2))], # Chargeability
        [np.zeros((nlayer-1, nlayer_sum*2)), Wx_block, np.zeros((nlayer-1, nlayer_sum*1))], # Time constant
        [np.zeros((nlayer-1, nlayer_sum*3)), Wx_block], # Exponent C
        ])
        self.Wx = Wx
        return Wx

    def get_Wx_r(self):
        nlayer = self.sim.nlayer
        if nlayer == 1:
            print("No smoothness for one layer model")
            nM = self.nM
            nP = self.nP
            Wx = np.zeros((nM,nM))
            self.Wx = Wx
            return Wx
        depth = self.sim.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        x = (depth[:-1] + depth[1:]) / 2
        delta_x = np.diff(x)
        elm1 = 1/delta_x
        elm2 = np.sqrt(delta_x)
        Wx_block = np.zeros((nlayer-1, nlayer))
        Wx_block[:,:-1] = -np.diag(elm2*elm1)
        Wx_block[:,1:] += np.diag(elm2*elm1)
        Wx=np.block([
        [Wx_block], # Resistivity
        ])
        self.Wx = Wx
        return Wx

    def get_Wx_rm(self):
        nlayer = self.sim.nlayer
        nM_r = self.sim.nM_r
        nM_m = self.sim.nM_m
        nM_t = self.sim.nM_t
        nM_c = self.sim.nM_c
        depth = self.sim.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        x = (depth[:-1] + depth[1:]) / 2

        if nlayer == 1:
            print("No smoothness for one layer model")
            nM = self.sim.nM
            nP = self.sim.nP
            Wx = np.zeros((nM,nP))
            self.Wx = Wx
            return Wx
        delta_x = np.diff(x)
        elm1 = 1/delta_x
        elm2 = np.sqrt(delta_x)
        Wx_block = np.zeros((nlayer-1, nlayer))
        Wx_block[:,:-1] = -np.diag(elm2*elm1)
        Wx_block[:,1:] += np.diag(elm2*elm1)
        Wx=np.block([
        [Wx_block, np.zeros((nM_r-1, nM_m + nM_t + nM_c ))], # Resistivity
        [np.zeros((nM_m-1, nM_r)), Wx_block, np.zeros((nM_m-1,  nM_t + nM_c ))], # Chargeability
        ])
        self.Wx = Wx
        return Wx
    
    def get_Ws_sea_one_tau_c(self):
        nlayer = self.sim.nlayer
        nM_r = self.sim.nM_r
        nM_m = self.sim.nM_m
        nM_t = self.sim.nM_t
        nM_c = self.sim.nM_c
        depth = self.sim.model_base["depth"]
        depth= np.r_[depth,2*depth[-1]-depth[-2]]
        delta_x = np.diff(depth)
        elm1 =  np.sqrt(delta_x)
        Ws_block1 = np.diag(elm1)
        Ws_block2=  np.r_[elm1.sum()]
        Ws = np.block([
            [Ws_block1, np.zeros((nlayer, nM_m)),np.zeros((nlayer,nM_t+nM_c))], # Resistivity
            [np.zeros((nlayer, nM_r)), Ws_block1,np.zeros((nlayer,nM_t+nM_c))], # Chargeabillity
            [np.zeros((1, nM_r+nM_m)), Ws_block2,np.zeros((1,nM_c))], # time_constant
            [np.zeros((1, nM_r+nM_m+nM_t)), Ws_block2], # time_constant
        ])
        self.Ws = Ws
        return Ws    

    def loss_func(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * np.dot(r,r)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim += 0.5 * self.alphas*np.dot(rms, rms)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*np.dot(rmx, rmx)
        return phid+beta*phim, phid, phim
    
    def loss_func_L2(self,m, beta, gradient=False,update_Wsen=False,m_ref=None):
        if gradient:
            J = self.J(m)
            if update_Wsen:
                self.update_Ws(J)
            dpred_m = self.dpred(m)            
            rd = self.Wd @ (dpred_m -self.dobs)
            phid = 0.5 * np.dot(rd,rd)
            g = J.T @ self.Wd.T@ rd
            H = J.T @ self.Wd.T@ self.Wd@J
            phim = 0
            if m_ref is not None:
                rms = self.Ws @ (m - m_ref)
                phim += 0.5 * self.alphas*np.dot(rms, rms)
                g += beta * self.alphas * (self.Ws.T@self.Ws@ (m - m_ref))
                H += beta * self.alphas * self.Ws.T@self.Ws
            if self.Wx is not None:
                rmx = self.Wx @ m
                phim += 0.5 * self.alphax*np.dot(rmx, rmx)
                g += beta * self.alphax * (
                    self.Wx.T @ self.Wx @ m)
                H += beta * self.alphax * self.Wx.T @ self.Wx
            f= phid+beta*phim
            return f, phid, phim, dpred_m, g, H
        else:
            dpred_m = self.dpred(m)            
            rd = self.Wd @ (dpred_m-self.dobs)
            phid = 0.5 * np.dot(rd,rd)
            phim = 0
            if m_ref is not None:
                rms = self.Ws @ (m - m_ref)
                phim += 0.5 * self.alphas*np.dot(rms, rms)
            if self.Wx is not None:
                rmx = self.Wx @ m
                phim += 0.5 * self.alphax*np.dot(rmx, rmx)
            f= phid+beta*phim
            return f, phid, phim, dpred_m

    def BetaEstimate_byEig(self,mvec, beta0_ratio=1.0, update_Wsen=False):
        J = self.J(mvec)

        if update_Wsen:
            self.update_Ws(J)            

        # Effective data misfit term with projection matrix
        A_data =  J.T @ self.Wd.T @ self.Wd @ J 
        eig_data = np.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        # A_reg = alphax* Prj_m.T @ self.Wx.T @ self.Wx @ Prj_m
        A_reg = np.zeros_like(A_data)
        if self.Wx is not None:
            A_reg += self.alphax * self.Wx.T @ self.Wx 
        if self.Ws is not None:
            A_reg += self.alphas * (self.Ws.T @ self.Ws)
        eig_reg = np.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > self.eig_tol]
        eig_reg = eig_reg[eig_reg > self.eig_tol]

        # Use the ratio of eigenvalues to set beta range
        lambda_d = np.max(eig_data)
        lambda_r = np.min(eig_reg)
        return beta0_ratio * lambda_d / lambda_r

    def steepest_descent(self, dobs, model_init, niter):
        '''
        Eldad Haber, EOSC555, 2023, UBC-EOAS 
        '''
        model_vector = model_init
        r = dobs - self.predicted_data(model_vector)
        f = 0.5 * np.dot(r, r)

        error = np.zeros(niter + 1)
        error[0] = f
        model_itr = np.zeros((niter + 1, model_vector.shape[0]))
        model_itr[0, :] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.J(model_vector)
            r = dobs - self.dpred(model_vector)
            dm = J.T @ r
            g = np.dot(J.T, r)
            Ag = J @ g
            alpha = np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector = self.project_convex_set(model_vector + alpha * dm)
            r = self.dpred(model_vector) - dobs
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(dm) < 1e-12:
                break
            error[i + 1] = f
            model_itr[i + 1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
        return model_vector, error, model_itr

    def Gradient_Descent(self, dobs, mvec_init, niter, beta, alphas, alphax,
            s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4, ELS=True, BLS=True ):
        mvec_old = mvec_init
        mvec_new = None
        mref = mvec_init
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        rd = self.Wd @ (self.dpred(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(self.Ws@(mvec_old - mref), self.Ws@(mvec_old - mref))
        rmx = 0.5 * np.dot(self.Wx @ mvec_old, self.Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        k = 0
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old
        print(f'Gradient Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.J(mvec_old)
            g = J.T @ self.Wd.T @ rd + beta * (alphas * self.Ws.T @ self.Ws @ (mvec_old - mref)
                                          + alphax * self.Wx.T @ self.Wx @ mvec_old)

            # Exact line search
            if ELS:
                t = np.dot(g,g)/np.dot(self.Wd@J@g,self.Wd@J@g)
#                t = (g.T@g)/(g.T@J.T@J@g)
            else:
                t = 1.

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # Line search method Armijo using directional derivative
            s = s0
            dm = t*g
            directional_derivative = np.dot(g, -dm)

            mvec_new = self.project_convex_set(mvec_old - s * dm)
            rd = self.Wd @ (self.dpred(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(self.Ws @ (mvec_new - mref), self.Ws @ (mvec_new - mref))
#            rmx = 0.5 * np.dot(self.Wx @ mvec_new, self.Wx @ mvec_new)
            rmx = 0.5 * np.dot(self.Wx @ mvec_new, self.Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim
            if BLS:
                while f_new >= f_old + s * mu * directional_derivative:
                    s *= sfac
                    mvec_new = self.project_convex_set(mvec_old - s * dm)
                    rd = self.Wd @ (self.dpred(mvec_new) - dobs)
                    phid = 0.5 * np.dot(rd, rd)
                    rms = 0.5 * np.dot(self.Ws @ (mvec_new - mref), self.Ws @ (mvec_new - mref))
                    rmx = 0.5 * np.dot(self.Wx @ mvec_new, self.Wx @ mvec_new)
                    phim = alphas * rms + alphax * rmx
                    f_new = phid + beta * phim
                    if np.linalg.norm(s) < stol:
                        break
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, s:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # filter model prog data
        mvec_prg = mvec_prg[:k]
        error_prg = error_prg[:k]
        # Save Jacobian
        self.Jacobian = J
        return mvec_new, error_prg, mvec_prg

    def GaussNewton(self,mvec_init, niter, beta0, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac=0.5,update_Wsen=False,
        stol=1e-6, gtol=1e-3, mu=1e-4):
        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        self.betas = []

        mvec_old = mvec_init.copy()
        m_ref = mvec_init
        beta= beta0
        f_old, phid, phim = self.loss_func(mvec_old,beta, m_ref=m_ref)# phid

        for i in range(niter):
            dpred_old = self.dpred(mvec_old)
            self.betas.append(beta)
            self.error_prg.append([f_old, phid, phim])
            self.mvec_prg.append(mvec_old)
            self.data_prg.append(dpred_old)

            beta = beta0 / (coolingFactor ** (i // coolingRate))
            rd = self.Wd@(dpred_old - self.dobs)
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

            dm = np.linalg.solve(H, g)  # Ensure dm is a 1D tensor
            g_norm = np.linalg.norm(g, ord=2)

            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break
            s = s0
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            f_new, phid, phim = self.loss_func(mvec_new,beta, m_ref=m_ref)# phid
            directional_derivative = np.dot(g.flatten(), -dm.flatten())
            while f_new >= f_old + s * mu * directional_derivative:
                s *= sfac
                mvec_new = self.project_convex_set(mvec_old - s * dm)
                f_new, phid, phim = self.loss_func(mvec_new,beta,m_ref=m_ref) #phid
                if s < stol:
                    break
            mvec_old = mvec_new
            f_old = f_new
            if print_update:
                print(f'{i+1:3}, beta:{beta:.1e}, step:{s:.1e}, g:{g_norm:.1e}, phid:{phid:.1e}, phim:{phim:.1e}, f:{f_new:.1e} ')

        self.error_prg.append([f_new, phid, phim])
        self.mvec_prg.append(mvec_new)
        self.data_prg.append(self.dpred(mvec_new))
        self.betas.append(beta)

        return mvec_new


class InducedPolarization:

    def __init__(self,
        res0=None, con8=None, eta=None, tau=None, c=None,
        freq=None, times=None, windows_strt=None, windows_end=None
        ):

        if res0 is not None and con8 is not None and eta is not None:
            assert np.allclose(con8 * res0 * (1 - eta), 1.)
        self.con8 = con8
        self.res0 = res0
        self.eta = eta
        if self.res0 is None and self.con8 is not None and self.eta is not None:
            self.res0 = 1./ (self.con8 * (1. - self.eta))
        if self.res0 is not None and self.con8 is None and self.eta is not None:
            self.con8 = 1./ (self.res0 * (1. - self.eta))
        self.tau = tau
        self.c = c
        self.freq = freq
        self.times = times
        self.windows_strt = windows_strt
        self.windows_end = windows_end

    def validate_times(self, times):
        assert np.all(times >= -eps ), "All time values must be non-negative."
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def pelton_res_f(self, freq=None, res0=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return res0*(1.-eta*(1.-1./(1. + iwtc)))

    def pelton_con_f(self, freq=None, con8=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return con8-con8*(eta/(1.+(1.-eta)*iwtc))

    def debye_con_t(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        ind_0 = (times == 0)
        debye[ind_0] = 1.0
        debye[~ind_0] = -eta/((1.0-eta)*tau)*np.exp(-times[~ind_0]/((1.0-eta)*tau))
        return con8*debye

    def debye_con_t_intg(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        return con8 *(1.0 -eta*(1. -np.exp(-times/((1.0-eta)*tau))))

    def debye_res_t(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        res8 = res0 * (1.0 - eta)
        ind_0 = (times == 0)
        debye[ind_0] = res8 
        debye[~ind_0] = (res0-res8)/tau * np.exp(-times[~ind_0] / tau)
        return debye

    def debye_res_t_intg(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        res8 = res0 * (1.0 - eta)
        return res8 + (res8 - res0)*(np.exp(-times/tau) - 1.0)

    def freq_symmetric(self,f):
        symmetric = np.zeros_like(f, dtype=complex)
        nstep = len(f)
        half_step = nstep // 2
        if nstep % 2 == 0:
            symmetric[:half_step] = f[:half_step]
            symmetric[half_step] = f[half_step].real
            symmetric[half_step+1:] = f[1:half_step].conj()[::-1]
        else:
            symmetric[:half_step+1] = f[:half_step+1]
            symmetric[half_step+1:] = f[1:half_step].conj()[::-1]

        # assert np.allclose(symmetric[:half_step].real, symmetric[half_step:].real[::-1])
        # assert np.allclose(symmetric[:half_step].imag, -symmetric[half_step:].imag[::-1])
        return symmetric

    def get_frequency_tau(self, tau=None, log2nfreq=16): 
        tau = self.get_param(tau, self.tau)
        log2nfreq = int(log2nfreq)
        nfreq = 2**log2nfreq
        freqcen = 1 / tau
        freqend = freqcen * nfreq**0.5
        freqstep = freqend / nfreq
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq

    def get_frequency_tau2(self, tau=None, log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        freqcen = 1 / tau
        freqend = freqcen * 2**log2max
        freqstep = freqcen * 2**log2min
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq


    def get_frequency_tau_times(self, tau=None, times=None,log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        times = self.get_param(times, self.times)
        self.validate_times(times)
        _, windows_end = self.get_windows(times)

        freqstep = 1/tau*(2**np.floor(np.min(
            np.r_[log2min,np.log2(tau/windows_end[-1])]
        )))
        freqend = 1/tau*(2**np.ceil(np.max(
            np.r_[log2max, np.log2(2*tau/min(np.diff(times)))]
        )))
        freq = np.arange(0,freqend,freqstep)
        self.freq=freq
        print(f'log2(freq) {np.log2(len(freq))} considering tau and times')
        return freq

    def compute_fft(self, f):
        f_sym = self.freq_symmetric(f)
        return np.fft.ifft(f_sym)


    def pelton_fft(self, con_form=True, con8=None, res0=None, eta=None, tau=None, c=None, freq=None):
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c) 
        freq = self.get_param(freq, self.freq) 
        freqstep = freq[1] - freq[0]
        freqend = freq[-1] +freqstep

        if con_form:
            con8 = self.get_param(con8, self.con8)
            fft_f = self.pelton_con_f(freq=freq,
                     con8=con8, eta=eta, tau=tau, c=c)
        else:
            res0 = self.get_param(res0, self.res0)
            fft_f = self.pelton_res_f(freq=freq,
                     res0=res0, eta=eta, tau=tau, c=c)
        fft_times, fft_data = self.compute_fft(fft_f, freqend, freqstep)
        return fft_times, fft_data

    def get_windows(self, times):
        self.validate_times(times)
        windows_strt = np.zeros_like(times)
        windows_end = np.zeros_like(times)
        dt = np.diff(times)
        windows_strt[1:] = times[:-1] + dt / 2
        windows_end[:-1] = times[1:] - dt / 2
        windows_strt[0] = times[0] - dt[0] / 2
        windows_end[-1] = times[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def apply_windows(self, times, data, windows_strt=None, windows_end=None):
        if windows_strt is None:
            windows_strt = self.windows_strt
        if windows_end is None:
            windows_end = self.windows_end
        self.validate_times(times)

        # Find bin indices for start and end of each window
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        window_data = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                window_data[i] = np.mean(data[start:end])

        return window_data

    def get_window_matrix (self, times, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)
        nwindows = len(windows_strt)
        window_matrix = np.zeros((nwindows, len(times)))
        for i in range(nwindows):
            start = windows_strt[i]
            end = windows_end[i]
            ind_time = (times >= start) & (times <= end)
            if ind_time.sum() > 0:
                window_matrix[i, ind_time] = 1/ind_time.sum()
        return window_matrix    
    

class TEM_Signal_Process:
    
    def __init__(self,  
        base_freq,on_time, rmp_time, rec_time, smp_freq,
        windows_cen=None, windows_strt = None, windows_end = None):
        self.base_freq = base_freq
        self.on_time = on_time
        self.rmp_time = rmp_time
        self.rec_time = rec_time
        self.smp_freq = smp_freq
        time_step = 1./smp_freq
        self.time_step = time_step
        self.times_rec = np.arange(0,rec_time,time_step) + time_step
        self.times_filt = np.arange(0,rec_time,time_step)
        self.windows_cen= windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
    
    def get_param(self, param, default):
        return param if param is not None else default

    def validate_times(self, times):
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def get_windows_cen(self, windows_cen):
        self.validate_times( windows_cen)
        self.windows_cen = windows_cen
        windows_strt = np.zeros_like( windows_cen)
        windows_end = np.zeros_like( windows_cen)
        dt = np.diff( windows_cen) + 2*eps
        windows_strt[1:] =  windows_cen[:-1] + dt / 2
        windows_end[:-1] =  windows_cen[1:] - dt / 2
        windows_strt[0] =  windows_cen[0] - dt[0] / 2
        windows_end[-1] =  windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def get_window_linlog(self,linstep,time_trns):
        rmp_time = self.rmp_time
        rec_time = self.rec_time + rmp_time
        nlinstep = round(time_trns/linstep)
        logstep = np.log((linstep+time_trns)/time_trns)
        logstrt = np.log(time_trns)
#        logend = np.log(rec_time) + logstep + eps
        logend = np.log(rec_time) - eps
        nlogstep = round((logend-logstrt)/logstep)
        windows_cen= np.r_[np.arange(0,time_trns,linstep), np.exp(np.arange(logstrt,logend,logstep))]
        windows_strt = np.r_[np.arange(0,time_trns,linstep)-linstep/2, np.exp(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))]
        windows_end =  np.r_[np.arange(0,time_trns,linstep)+linstep/2,  np.exp(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))]
        self.windows_cen = windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        print(f'linear step: {nlinstep}, log step: {nlogstep}, total steps: {nlinstep+nlogstep}')
        return windows_cen, windows_strt, windows_end

    def get_window_log(self,logstep, tstart, tend=None, rmp_time=None):
        tend = self.get_param(tend, self.rec_time)
        rmp_time = self.get_param(rmp_time, self.rmp_time)
        if rmp_time is not None:
            tstart = tstart
            tend = tend - rmp_time     

        logstrt = np.log10(tstart)
        logend = np.log10(tend)
        log10_windows_cen = np.arange(logstrt,logend,logstep)
#        log10_windows_cen = np.linspace(logstrt,logend,logstep)
        windows_cen  = 10.**log10_windows_cen +rmp_time
        windows_strt = 10.**(log10_windows_cen-logstep/2) +rmp_time
        windows_end  = 10.**(log10_windows_cen+logstep/2) +rmp_time
        # windows_strt = 10.**(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))
        # windows_end  = 10.**(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))
        # print(f'log step: {len(self.windows_cen_log)}')
        self.windows_cen = windows_cen 
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_cen, windows_strt, windows_end

    def window(self,times,data, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)

        # Find bin indices for start and end of each windows
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        data_window = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                data_window[i] = np.mean(data[start:end])
        return data_window
    
    def get_window_matrix (self, times, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)
        nwindows = len(windows_strt)
        window_matrix = np.zeros((nwindows, len(times)))
        for i in range(nwindows):
            start = windows_strt[i]
            end = windows_end[i]
            ind_time = (times >= start) & (times <= end)
            if ind_time.sum() > 0:
                window_matrix[i, ind_time] = 1/ind_time.sum()
        return window_matrix

    def plot_window_data(self,data=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        windows_strt= self.windows_strt
        windows_end = self.windows_end
        windows_cen = self.windows_cen
        if data is None:
            ax.loglog(windows_cen, windows_cen,"k*")
            ax.loglog(windows_strt, windows_cen,"b|")
            ax.loglog(windows_end, windows_cen,"m|")
        else:
            assert len(data) == len(self.windows_cen), "Data and windows must have the same length."
            ax.loglog(windows_cen, data,"k*")
            ax.loglog(windows_strt, data,"b|")
            ax.loglog(windows_end, data,"m|")
        ax.grid(True, which="both")
        ax.legend(["center","start","end"])
        return ax

    def butter_lowpass(self, cutoff, order=1):
        fs = self.smp_freq
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
   
    def apply_lowpass_filter(self, data, cutoff, order=1):
        b, a = self.butter_lowpass(cutoff, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def filter_linear_rmp(self, rmp_time=None, times_rec=None, time_step=None):
        rmp_time  = self.get_param(rmp_time, self.rmp_time)
        times_rec = self.get_param(times_rec, self.times_rec)
        time_step = self.get_param(time_step, self.time_step)
        times_filt = self.times_filt
        filter_linrmp = np.zeros_like(times_filt)
        inds_rmp = times_filt <= rmp_time
        filter_linrmp[inds_rmp] =   1.0/float(inds_rmp.sum())
        return filter_linrmp

    def filter_linear_rmp_rect(self, rmp_time=None):
        if rmp_time is None:
            rmp_time = self.rmp_time
        pos_off = self.filter_linear_rmp(rmp_time=rmp_time)
        return np.r_[-pos_off, pos_off]
        
    def rect_wave(self, t, base_freq=None, neg=False):
        self.get_param(base_freq, self.base_freq)
        if neg:
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.25))
            neg=-0.5*(1.0+signal.square(2*np.pi*(base_freq*t+0.5),duty=0.25))
            return pos + neg
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            return pos

    def rect_wave_rmp(self, t, base_freq=None, rmp_time=None,neg=False):
        self.get_param(base_freq, self.base_freq)
        self.get_param(rmp_time, self.rmp_time)
        if neg:
            print("under construction")
            return None
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            ind_pos_on = t<=rmp_time
            pos[ind_pos_on] = t/rmp_time
            ind_pos_off = (t>=0.5/base_freq) & (t<=0.5/base_freq+rmp_time)
            pos[ind_pos_off] = 1.0 - (t-0.5/base_freq)/rmp_time
            return pos


    def interpolate_data_lin(self,times,data, times_rec=None,method='cubic'):
        '''
        times (array-like): Original time points (not uniformly spaced).
        data (array-like): Original data values at time points `t`.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        Returns:
            resampled_data (np.ndarray): Resampled data on `t_new`.
        '''
        times_rec = self.get_param(times_rec, self.times_rec)
        interpolator = interp1d(
            x=times,
            y=data,
            kind=method,
            fill_value='extrapolate'
        )
        return interpolator(times_rec)


    def interpolate_data(self,times,data, times_rec=None,method='cubic',
        logmin_time=1e-8, linScale_time=1.0, logmin_data=1e-8, linScale_data=1.0):
        '''
        times (array-like): Original time points (not uniformly spaced).
        data (array-like): Original data values at time points `t`.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        Returns:
            resampled_data (np.ndarray): Resampled data on `t_new`.
        '''
        times_rec = self.get_param(times_rec, self.times_rec)
        pslog_time = PsuedoLog(logmin=logmin_time, linScale=linScale_time)
        pslog_data = PsuedoLog(logmin=logmin_data, linScale=linScale_data)
        if method == "linear":
            interpolator = interp1d(
                x=pslog_time.pl_value(times),
                y=pslog_data.pl_value(data),
                kind=method,
                fill_value='extrapolate'
            )
        if method == "cubic":
            interpolator = CubicSpline(
                x=pslog_time.pl_value(times),
                y=pslog_data.pl_value(data),
            )
        
        return pslog_data.pl_to_linear(interpolator(pslog_time.pl_value(times_rec)))

    def deconvolve(self, data, data_pulse):
        filt, reminder = signal.deconvolve(
            np.r_[data, np.zeros(len(data)-1)],
            data_pulse
            )
        print(reminder)
        print(np.linalg.norm(reminder))
        return filt

class PsuedoLog:
    def __init__(self, logmin=None, linScale=None, max_y=eps, min_y=-eps,
        logminx=None, linScalex=None, max_x=eps, min_x=-eps):
        self.logmin = logmin
        self.linScale = linScale
        self.max_y = max_y
        self.min_y = min_y
        self.logminx = logminx
        self.linScalex = linScalex
        self.max_x = max_x
        self.min_x =min_x

    def get_param(self, param, default):
        return param if param is not None else default

    def pl_value(self, lin, logmin=None, linScale=None):    
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(lin)
        if is_scalar:
            lin = np.array([lin])  # Convert scalar to array for uniform processing
                
        abs_lin = np.abs(lin)
        sign_lin = np.sign(lin)
        ind_pl = (abs_lin >= logmin)
        ind_lin = ~ind_pl
        plog = np.zeros_like(lin)
        plog[ind_pl] = sign_lin[ind_pl] * (
            np.log10(abs_lin[ind_pl] / logmin) + linScale
            )
        plog[ind_lin] = lin[ind_lin] / logmin * linScale
        return plog
    
    def pl_to_linear(self,plog, logmin=None, linScale=None):   
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(plog)
        if is_scalar:
            lin = np.array([plog])  # Convert scalar to array for uniform processing
        abs_plog = np.abs(plog)
        sign_plog = np.sign(plog)
        ind_pl = (abs_plog >= linScale)
        ind_lin = ~ind_pl
        lin = np.zeros_like(plog)
        lin[ind_pl] = sign_plog[ind_pl] * logmin * 10 ** (abs_plog[ind_pl] - linScale)
        lin[ind_lin] = plog[ind_lin] / linScale * logmin
        return lin

    def semiply(self, x, y, logmin=None, linScale=None, ax=None, xscale_log=True,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        logmin = self.get_param(logmin, self.logmin)
        linScale = self.get_param(linScale, self.linScale)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)
        
        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        if xscale_log:
            ax.semilogx(x, plog_y, **default_kwargs)
        else:
            ax.plot(x, plog_y, **default_kwargs)
        
        self.max_y = np.max(np.r_[self.max_y,np.max(y)])
        self.min_y = np.min(np.r_[self.min_y,np.min(y)])
        return ax


    def semiplx(self, x, y,logminx=None,linScalex=None,ax=None, yscale_log=True,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        if yscale_log:
            ax.semilogy(plog_x, y, **default_kwargs)
        else:
            ax.plot(plog_x, y, **default_kwargs)
        self.max_x = np.max(np.r_[self.max_x,np.max(x)])
        self.min_x = np.min(np.r_[self.min_x,np.min(x)])
        return ax

    def plpl_plot(self, x, y,
        logminx=None,linScalex=None,logmin=None,linScale=None,ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        logminx = self.get_param(logminx, self.logminx)
        linScalex = self.get_param(linScalex, self.linScalex)
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
        }
        default_kwargs.update(kwargs)
        ax.plot(plog_x, plog_y, **default_kwargs)
        self.max_y = np.max(np.r_[self.max_y,np.max(y)])
        self.min_y = np.min(np.r_[self.min_y,np.min(y)])
        self.max_x = np.max(np.r_[self.max_x,np.max(x)])
        self.min_x = np.min(np.r_[self.min_x,np.min(x)])
        return ax

    def pl_axes(self,ax,logmin=None,linScale=None,max_y=None,min_y=None):
        assert hasattr(ax, 'set_xlim') and hasattr(ax, 'set_xticks') and hasattr(ax, 'set_xticklabels'), \
        "Provided 'ax' is not a valid Matplotlib Axes object."
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        max_y = self.get_param(max_y, self.max_y)
        min_y= self.get_param(min_y, self.min_y)

        if max_y <= logmin:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10((max_y+eps)/logmin)+1))
        posticks = linScale + np.arange(n_postick)
        #poslabels = logmin*10**np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logmin * 10**np.arange(n_postick))]

        if -min_y <= logmin:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10((-min_y+eps)/logmin)+1))

        negticks = -linScale - np.arange(n_negtick)
        negticks = negticks[::-1]
        #neglabels = -logmin*10**np.arange(n_negtick)
        neglabels = [f"{v:.0e}" for v in (-logmin * 10**np.arange(n_negtick))[::-1]]
#        neglabels = neglabels[::-1]
#        ticks  = np.hstack(( negticks, [0], posticks))
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_ylim([min(ticks), max(ticks)])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        # reset max and min
        self.max_y = eps
        self.min_y = -eps
        return ax

    def pl_axes_x(self,ax,logminx=None,linScalex=None,max_x=None,min_x=None):
        assert hasattr(ax, 'set_xlim') and hasattr(ax, 'set_xticks') and hasattr(ax, 'set_xticklabels'), \
        "Provided 'ax' is not a valid Matplotlib Axes object."
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        max_x = self.get_param(max_x, self.max_x)
        min_x= self.get_param(min_x, self.min_x)
        if max_x <= logminx:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10(max_x/logminx)+1))
        posticks = linScalex + np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logminx * 10**np.arange(n_postick))]
        if -min_x <= logminx:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10(-min_x/logminx)+1))
        negticks = -linScalex - np.arange(n_negtick)
        negticks = negticks[::-1]
        neglabels = [f"{v:.0e}" for v in (-logminx * 10**np.arange(n_negtick))[::-1]]
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_xlim([min(ticks), max(ticks)])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        # reset max and min
        self.max_x = eps
        self.min_x = -eps
        return ax
    
    def pl_axvline(self, ax, x, **kwargs):
        logminx = self.logminx
        linScalex = self.linScalex
        default_kwargs = {
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1.0,
        }
        default_kwargs.update(kwargs)
        ax.axvline(self.pl_value(x,logmin=logminx, linScale=linScalex), **default_kwargs)
        return ax
    
    def pl_axhline(self, ax, y, **kwargs):
        logmin = self.logmin
        linScale = self.linScale
        default_kwargs = {
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1.0,
        }
        default_kwargs.update(kwargs)
        ax.axhline(self.pl_value(y, logmin=logmin,linScale=linScale), **default_kwargs)
        return ax

def solve_polynomial(a, n,pmax):
    # Coefficients of the polynomial -x^{n+1} + (1+a)x - a = 0
    coeffs = [-1] + [0] * (n-1) + [(1 + a), -a]  # [-1, 0, ..., 0, (1 + a), -a]
    
    # Find the roots of the polynomial
    roots = np.roots(coeffs)
    
    # Filter real roots
    real_roots = [r.real for r in roots if np.isreal(r)]
    
    # Find the real root closest to pmax
    if real_roots:
        closest_root = real_roots[np.argmin(np.abs(np.array(real_roots) - pmax))]
        return closest_root
    else:
        return None  # Return None if no real roots are found

def mesh_Pressure_Vessel(tx_radius,cs1,ncs1, pad1max,cs2,max,lim,pad2max): 
    h1a = discretize.utils.unpack_widths([(cs1, ncs1)])
    a1 = (tx_radius- np.sum(h1a))/cs1 
    n_tmp = -1 + np.log((a1+1)*pad1max-a1)/np.log(pad1max)
    npad1b= int(np.ceil(n_tmp))
    pad1 = solve_polynomial(a1, npad1b, pad1max)
    npad1c = int(np.floor(np.log(cs2/cs1)/np.log(pad1))-npad1b)
    if npad1c< 0:
        print("error: padx1max is too large")

    h1bc = discretize.utils.unpack_widths([(cs1, npad1b+npad1c, pad1)])

    ncs2 = int(np.ceil( (max-np.sum(np.r_[h1a,h1bc])) / cs2 ))

    h2a= discretize.utils.unpack_widths([(cs2, ncs2)])

    a2 = (lim-np.sum(np.r_[h1a, h1bc, h2a]))/cs2 
    n_tmp = -1 + np.log((a2+1)*pad2max-a2)/np.log(pad2max)
    npad2 = int(np.ceil(n_tmp))
    pad2 = solve_polynomial(a2, npad2, pad2max)
    h2b = discretize.utils.unpack_widths([(cs2, npad2, pad2)])
    h = np.r_[h1a,h1bc,h2a,h2b]
    return h
