import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
from sklearn import preprocessing
from numpy.linalg import inv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward(x, dx):
    
    sample = fftshift(fft2(ifftshift(x)))
    sample = sample*np.power(dx, 2)
    
    return sample

def forwardtrans(x, dx):
    
    sample = fftshift(ifft2(ifftshift(x)))
    sample = sample/np.power(dx, 2)
    
    return sample

def fft2c(x):
    '''
    Centered fft applies fft to last 2 axes by default
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft applies fft to last 2 axes by default
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res

def fourier_matrix(rows, cols):
    '''
    return unitary (rows x cols) fourier matrix
    '''
    # from scipy.linalg import dft
    # return dft(rows,scale='sqrtn')

    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fourier_matrix = np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale

    return fourier_matrix


def inverse_fourier_matrix(rows, cols):
    return np.array(np.matrix(fourier_matrix(rows, cols)).getH())

def ismember(d, k):
    return [1 if (i == k) else 0 for i in d]

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def x_id(x):
    
    TEST = np.exp(1j*x)
    E0 = TEST
    I_0 = np.power(np.abs(E0),2)
    
    return I_0

def x_init(x):
    
    TEST = np.exp(1j*x)
    
    return TEST
    

def deri(x, **kwargs):
    if len(kwargs.items()) < 3:
        if x.shape[0] == 1:
            dim = 1
        else:
            dim = 0
    else:
        if np.isscalar(dim) == 0 or ismember(dim, [1,2]) == 0:
            print('dim must be 1 or 2')
            
    if len(kwargs.items()) < 2 or len(N) == 0:
        N = 1
    else:
        if np.isscalar(N) == 0 or N != round(N):
            print('N must be a scalar integer!')
    
    dx = x
    for n in range(N):
        dif = np.diff(dx, 1, dim)
        
        if dim == 0:
            first = np.vstack((dif[0,...],dif))
            last = np.vstack((dif, dif[dif.shape[0]-1,...]))
            
        elif dim == 1:
            first = np.vstack((dif[...,0], dif))
            last = np.vstack((dif, dif[...,dif.shape[0]-1]))
        
        dx = (first + last) / 2
        
    return dx

class fresnel_prop(nn.Module):
    def __init__(self, dpix, _lambda, Hsize, method, batch = 1, **kwargs):
        super(fresnel_prop, self).__init__()
        self.dpix = dpix
        self._lambda = _lambda
        self.Hsize = Hsize
        self.method = method
        self.batch = batch
        self.low_bound = 0
        self.upper_bound = 1
        self.level = 100
        
    def forward(self, *input):
        return self.perform(*input)

    def addgaussiannoise(self, img, level):

        nx, ny = img.shape
        mean = 0
        var = 0.1
        sigma = var**level
        gauss = np.random.normal(mean,sigma,(nx, ny))
        gauss = gauss.reshape(nx, ny)
        noisy = img + gauss

        return noisy

    def perform(self, x, z):
        '''
        Parameters
        ------------------------------
        inputs: 1 4d tensors, the reconstructed phase x_i-1 from last PINN block
        four modes can be selected: cutoff, absorptive, fft and TIE
        Returns
        ------------------------------
        output: 4d tensor, data input with forward propagation H(φ) fresnel propagation + square of complex optical field
        '''
        
        # phase image preprocess and normalize the desired range 
        if torch.is_tensor(x) == True:
            with torch.no_grad():
                x = x.cpu().numpy()
        else:
            pass
        x = np.squeeze(x)
        minmax_scale = preprocessing.MinMaxScaler(feature_range=(self.low_bound, self.upper_bound))
        x = minmax_scale.fit_transform(x)
        fres_result = np.zeros((self.batch, 1, self.Hsize, self.Hsize))

        # create phase object
        T = np.ones(shape = x.shape)
        nx, ny = x.shape
        x = T * np.exp(1j*x)
        Htemp = np.fft.fft2(x)
        
        # define k space
        Xsize = self.Hsize*self.dpix
        du = 1/(Xsize)
        umax = 1/(2*self.dpix)

        fx = (np.concatenate((np.arange(-umax,0,du),np.arange(0,umax,du))))
        fy = (np.concatenate((torch.arange(0,nx/2),np.arange(-nx/2,0))))/nx
        U, V= np.meshgrid(fx,fx)
        #Circular window
        A = np.power(U, 2) + np.power(V, 2)
        
        #wave number
        k = 2*math.pi/self._lambda
        uev = 1/self._lambda
        if z == 0:
            unp = 1e10
        else:
            unp = uev*(Xsize/(2*abs(z)))
        #Ucut is correct   
        if uev >= unp:
            ucut = unp
        else:
            ucut = uev
 
        W = np.sqrt(A)
        W_shape = W.shape
        W_crop = np.zeros(W_shape)
        for x in range(0, W_shape[0]):
            for y in range(0, W_shape[1]):
                if W[x,y] <= ucut:
                    W_crop[x,y] = 1
                else:
                    W_crop[x,y] = 0

        # with interference
        if self.method == 'cutoff':       
            # paraxial approximation
            H = np.exp((-1j*math.pi*self._lambda* z)*(A))
            # Truncate kernel
            H = W_crop*H

        # without interference
        elif self.method == 'fft':
            H = np.exp(1j*k*z*(1-(np.power((self._lambda*U),2) + np.power((self._lambda*V),2))/2))

        elif self.method == 'TIE':
            I0 = x*np.conjugate(x)
            fx = np.fft.fftshift(U)
            fy = np.fft.fftshift(V)

            Cx = 2*1j*math.pi*fx
            Cy = 2*1j*math.pi*fy

            Fphi = np.fft.fft2(np.angle(x))
            Fdphidx = Fphi*Cx
            Fdphidy = Fphi*Cy
            dphidx = np.fft.ifft2(Fdphidx).real
            dphidy = np.fft.ifft2(Fdphidy).real

            dPsidx = I0*dphidx
            dPsidy = I0*dphidy

            FdPsidx = np.fft.fft2(dPsidx)
            FdPsidy = np.fft.fft2(dPsidy)

            Fd2Psidx2 = FdPsidx*Cx
            Fd2Psidy2 = FdPsidy*Cy

            d2Psidx2 = np.fft.ifft2(Fd2Psidx2).real
            d2Psidy2 = np.fft.ifft2(Fd2Psidy2).real

            laplacePsi = d2Psidx2+d2Psidy2
            dIdz = laplacePsi/(-k)
            fres_result = dIdz
            fres_result = torch.tensor(fres_result).to(device)
            fres_result = fres_result.type(torch.cuda.FloatTensor)
            
            # intensity derivative
            return fres_result
        
        HH = np.fft.fftshift(Htemp)
        RR = HH * H
        RR = np.fft.ifftshift(RR)
        R = np.fft.ifft2(RR)

        intensity = np.power(np.abs(R),2)
        
        fres_result[...] = self.addgaussiannoise(intensity,self.level)
        
        fres_result = torch.tensor(fres_result).to(device)
        fres_result = fres_result.type(torch.cuda.FloatTensor)
        
        return fres_result


class inverseTIE(nn.Module):
    def __init__(self, dpix, k, z, Hsize, batch = 1, **kwargs):
        super(inverseTIE, self).__init__()
        self.dpix = dpix
        self.z = z
        self.k= k
        self.Hsize = Hsize
        self.batch = batch
        self.norm = 100
        
    def forward(self, *input):
        return self.perform(*input)

    def perform(self, x_diff, x_i0):
        '''
        Parameters
        ------------------------------
        inputs: 2 4d tensors
            First is i0, second is the difference term of (H(φ)-i0)/Δz
        Returns
        ------------------------------
        output: 4d tensor, data input in iterative gradient term
        values
        '''
        if torch.is_tensor(x_diff) == True:
            with torch.no_grad():
                x_diff = x_diff.cpu().numpy()
                x_i0 = x_i0.cpu().numpy()
        else:
            pass

        x_diff = np.squeeze(x_diff)
        x_i0 = np.squeeze(x_i0)
        nx, ny = x_i0.shape
        p1_result = np.zeros((self.batch, 1, self.Hsize, self.Hsize))
        T0 = np.fft.fft2((self.k/(x_i0))*(x_diff))
        T1 = np.fft.fftshift(T0)
        
        # define k space
        Xsize = self.Hsize*self.dpix
        du = 1/Xsize
        umax = 1/(2*self.dpix)

        fx = (np.concatenate((np.arange(-umax,0,du),np.arange(0,umax,du))))
        fy = (np.concatenate((torch.arange(0,nx/2),np.arange(-nx/2,0))))/nx
        U, V= np.meshgrid(fx,fx)
        #Circular window
        A = np.power(U, 2) + np.power(V, 2)
        T2 = 4*math.pi*math.pi*A
        T2 = T2/(np.power(T2,2) + 1e-9) 
             
        RR = np.fft.ifftshift(T2*T1)
        P_uni = np.fft.ifft2(RR).real
        
        # p1_result[...] = P_uni / self.norm
        p1_result[...] = P_uni
        p1_result = torch.tensor(p1_result).to(device)
        p1_result = p1_result.type(torch.cuda.FloatTensor)
        
        return p1_result


class fresnel_prop_torch(nn.Module):
    def __init__(self, dpix, _lambda, Hsize, method, batch = 1, **kwargs):
        super(fresnel_prop_torch, self).__init__()
        self.dpix = dpix
        self._lambda = _lambda
        self.Hsize = Hsize
        self.method = method
        self.batch = batch
        # This can't be 0
        self.level = 100
        self.abs_deg = 25
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, *input):
        return self.perform(*input)

    def addgaussiannoise(self, img, level):

        nx, ny = img.shape
        mean = 0
        var = 0.1
        sigma = var**level
        gauss = np.random.normal(mean,sigma,(nx, ny))
        gauss = gauss.reshape(nx, ny)
        noisy = img + gauss

        return noisy

    def perform(self, x, z):
        '''
        All calculates taken on GPU
        Parameters
        ------------------------------
        inputs: 1 4d tensors, the reconstructed phase x_i-1 from last PINN block
        three modes can be selected: cutoff, fft and TIE
        Returns
        ------------------------------
        output: 4d tensor, data input with forward propagation H(φ) fresnel propagation + square of complex optical field
        '''
        
        # phase image preprocess and normalize the desired range 
        transforms.Normalize(0, 1)(x)
        batch,ch,nx,ny = [torch.tensor(i,device=self.device) for i in x.shape]
        pi,lamda,z,dpix,Hsize = [torch.tensor(i,device=self.device) for i in [np.pi,self._lambda,z,self.dpix,self.Hsize]]
        level,abs_deg = [torch.tensor(i,device=self.device) for i in [self.level,self.abs_deg]]
        
        # define k space
        Xsize = Hsize*dpix
        du = 1/(Xsize)
        umax = 1/(2*dpix)
        
        fx = (torch.cat((torch.arange(-umax,0,du),torch.arange(0,umax,du)))).to(self.device)
        # fy = (np.concatenate((torch.arange(0,nx/2),np.arange(-nx/2,0))))/nx
        U, V= torch.meshgrid(fx,fx)
        #Circular window
        A = U**2 + V**2
        

        # create phase object
        if self.method == 'cutoff' or self.method == 'fft':
            T = torch.ones(size = x.shape, device=self.device)
        else:
            x_abs = (torch.arange(-Hsize/2, Hsize/2)).to(self.device)*dpix
            y_abs = (torch.arange(-Hsize/2, Hsize/2)).to(self.device)*dpix
            X, Y = torch.meshgrid(x_abs,y_abs)
            A_abs = X**2 + Y**2
            T = torch.exp(-A_abs/dpix*abs_deg)
        x = torch.sqrt(T) * torch.exp(1j*x)
        Htemp = torch.fft.fft2(x)
        
        #wave number
        k = 2*pi/lamda
        uev = lamda
        if z == 0:
            unp = 1e10
        else:
            unp = uev*(Xsize/(2*torch.abs(z)))
        #Ucut is correct   
        if uev >= unp:
            ucut = unp
        else:
            ucut = uev
        
        # with interference
        if self.method == 'cutoff':
            W = (torch.sqrt(A)).cpu().numpy()
            W_shape = W.shape
            W_crop = np.zeros(W_shape)
            for x in range(0, W_shape[0]):
                for y in range(0, W_shape[1]):
                    if W[x,y] <= ucut:
                        W_crop[x,y] = 1
                    else:
                        W_crop[x,y] = 0
            W_crop = np.reshape(W_crop, (1,1,W_shape[0],W_shape[0]))
            W_crop = torch.tensor(W_crop,device = device)       
            # paraxial approximation
            H = torch.exp((-1j*pi*lamda*z)*(A))
            # Truncate kernel
            H = W_crop*H

        # without interference
        elif self.method == 'fft' or self.method == 'absorb':
            H = torch.exp(1j*k*z*((1-((lamda*U)**2) + ((lamda*V)**2)))/2)

        HH = torch.fft.fftshift(Htemp)
        RR = HH * H
        RR = torch.fft.ifftshift(RR)
        R = torch.fft.ifft2(RR)

        intensity = torch.abs(R)**2
        
        return intensity


class inverseTIE_torch(nn.Module):
    def __init__(self, dpix, k, z, Hsize, batch = 1, **kwargs):
        super(inverseTIE_torch, self).__init__()
        self.dpix = dpix
        self.z = z
        self.k= k
        self.Hsize = Hsize
        self.batch = batch
        self.method = 'fft'
        self.abs_deg = 25
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, *input):
        return self.perform(*input)

    def perform(self, x_diff, x_i0):
        '''
        All calculates take on GPU
        Parameters
        ------------------------------
        inputs: 2 4d tensors 
            First is i0, second is the difference term of (H(φ)-i0)/Δz
        Returns
        ------------------------------
        output: 4d tensor, data input in iterative gradient term
        values
        '''
        # phase image preprocess 
        batch,ch,nx,ny = [torch.tensor(i,device=self.device) for i in x_diff.shape]
        pi,z,dpix,Hsize,k,abs_deg = [torch.tensor(i,device=self.device) for i in [np.pi,self.z,self.dpix,self.Hsize,self.k,self.abs_deg]]

        if self.method == 'cutoff' or self.method == 'fft':
            T0 = torch.fft.fft2((k/(x_i0))*(x_diff))
        else:
            T0 = torch.fft.fft2((self.k/1)*(x_diff))
        T1 = torch.fft.fftshift(T0)
        
        # define k space
        Xsize = Hsize*dpix
        du = 1/Xsize
        umax = 1/(2*dpix)

        fx = (torch.cat((torch.arange(-umax,0,du),torch.arange(0,umax,du)))).to(self.device)
        # fy = (np.concatenate((torch.arange(0,nx/2),np.arange(-nx/2,0))))/nx
        U, V= torch.meshgrid(fx,fx)
        #Circular window
        A = U**2 + V**2

        T2 = 4*pi*pi*A
        T2 = T2/(T2**2 + 2e-52) 
             
        RR = torch.fft.ifftshift(T2*T1)
        P_uni = torch.real(torch.fft.ifft2(RR))

        if self.method == 'absorb':
            T = torch.exp(-A/dpix*abs_deg)
            Fx, Fy = torch.gradient(P_uni)
            Fxx,Fxy = torch.gradient (Fx/T)
            Fyx,Fyy = torch.gradient(Fy/T)
            DD = (Fxx+Fyy)/dpix**2

            T1 = torch.fft.fftshift((torch.fft.fft2(k*DD)))
            P_uni = (-torch.fft.ifft2(torch.fft.ifftshift(T2*T1)).real)*dpix/2
            
        return P_uni


class four_shotModel(nn.Module):
    def __init__(self,dpix, z, k, Hsize=512, batch=1, noiseCF=0.9,n_acc=4, **kwargs):
        super(four_shotModel, self).__init__()
        self.dpix = dpix
        self.z = z
        self.k = k
        self.Hsize = Hsize
        self.batch = batch
        self.n_acc = n_acc
        self.norm = 100
        
    def forward(self, *input):
        return self.perform(*input)

    def perform(self, x_stack,x_i0):
        '''
        Parameters of 4 shot TIE models in Transfer learning
        ------------------------------
        inputs: 2 4d tensors
            First is i0, second is the difference term from Gaussian TIE process
        Returns
        ------------------------------
        output: 4d tensor, data input in iterative gradient term
        values
        '''
        if torch.is_tensor(x_stack) == True:
            with torch.no_grad():
                x_stack = x_stack.cpu().numpy()
                x_i0 = x_i0.cpu().numpy()
        else:
            pass
        x_stack = np.squeeze(x_stack)
        n = int(self.n_acc/2)
        Deriv_num = 0
        Im = np.zeros((self.Hsize, self.Hsize))
        p1_result = np.zeros((self.batch, 1, self.Hsize, self.Hsize))
        minmax_scale = preprocessing.MinMaxScaler(feature_range=(-0.5, 1))

        # define k space
        Xsize = self.Hsize*self.dpix
        du = 1/Xsize
        umax = 1/(2*self.dpix)

        fx = (np.concatenate((np.arange(-umax,0,du),np.arange(0,umax,du))))
        # fy = (np.concatenate((torch.arange(0,nx/2),np.arange(-nx/2,0))))/nx
        U, V= np.meshgrid(fx,fx)
        #Circular window
        A = np.power(U, 2) + np.power(V, 2)

        rowv = np.arange(-n,n+1,1)
        M = np.zeros((2*n+1,2*n+1))
        MR = np.zeros((2*n+1,1))
        MR[1,:] = 1
        for i in range(2*n+1):
            M[i,:] = np.power(rowv,i)
        Min = inv(M)
        coeff = np.dot(Min,MR)

        
        for i in range(0,2*n+1):
            Im = x_stack[i,...]
            Deriv_num += Im*coeff[i]
        dIdZ = Deriv_num/self.z
        
        T0 = np.fft.fft2((self.k/1)*(dIdZ))
        T1 = np.fft.fftshift(T0)

        T2 = 4*math.pi*math.pi*A
        T2 = T2/(np.power(T2,2) + 2e-52) 

        P_ho = np.fft.ifft2(np.fft.ifftshift(T2*T1)).real
        p1_result[...] = minmax_scale.fit_transform(P_ho)
        p1_result = torch.tensor(p1_result).to(device)
        p1_result = p1_result.type(torch.cuda.FloatTensor)
        
        return p1_result
    
class Data_consistency(nn.Module):
    '''
    Data consistency layer
    '''
    def __init__(self, dpix, z, Hsize, _lambda, k, inv_op='2-shot', inv_noise_level=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'dc'

        super(Data_consistency, self).__init__()
        self.inv_noise_level = inv_noise_level
        self.inv_op = inv_op
        self.z = z
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # forward operator
        self.forward_operator = fresnel_prop_torch(dpix, _lambda, Hsize, method='fft', name='dc_fres')
        # inverse model
        if inv_op == '2-shot':
            self.inverse_fun = inverseTIE(dpix, k, z, Hsize, name='dc_2s_consistency')
        if inv_op == '4-shot':
            self.inverse_fun = four_shotModel(dpix, z, k, stack_level=2, name='dc_4s_consistency')

    def forward(self, *input):
        return self.perform(*input)

    def perform(self, x_phase, x_i0, x_id):
        #defouse distance for 2-shot TIE, operated on GPU only
        self.z = torch.tensor(self.z,device = self.device)
        
        # obtain defocus and infocus intensity measurements at different noise conditions
        I_posz = self.forward_operator.forward(x_phase,self.z)
        # x_result = I_posz
        I_infocus = self.forward_operator.forward(x_phase,0)
        I_negz = self.forward_operator.forward(x_phase,-self.z)
        
        # decide inverse function default: 2-shot TIE retrival, 4-shot is used at transfer learning
        if self.inv_op == '2-shot':
            I_diff = I_posz - I_negz / (2 * self.z)
            x_result = self.inverse_fun.forward(I_diff,I_infocus)
        #id stacks id±1,id±2
        if self.inv_op == '4-shot':
            I_pos4 = self.forward_operator.forward(x_phase, 4e-6)
            I_neg4 = self.forward_operator.forward(x_phase, -4e-6)
            I_diff = torch.cat((I_neg4, I_negz, I_infocus, I_posz, I_pos4), 1)
            x_result = self.inverse_fun.forward(I_diff,I_infocus)
        
        return x_result