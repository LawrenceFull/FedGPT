import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image
import math
import torch

# from tensorflow.python.ops.gen_array_ops import diag, transpose

np.random.seed(0)

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def converged(L, E, X, L_new, E_new):
    '''
    judge convered or not
    '''
    condition1 = np.max(abs(L_new - L))
    condition2 = np.max(abs(E_new - E))
    condition3 = np.max(abs(L_new + E_new - X))
    return max(condition1,condition2,condition3)

def SoftShrink(X, tau):
    '''
    apply soft thesholding
    '''
    z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)
    # z = np.clip(X-tau,a_min=0,a_max=None)+np.clip(X+tau,a_min=None,a_max=0)
    return z

def SVDShrink2(logger, Y, tau):
    '''
    apply tensor-SVD and soft thresholding
    '''

    [n1, n2, n3] = Y.shape
    XX = np.empty(shape = Y.shape, dtype=complex)
    X = np.fft.fft(Y)
    tnn = 0
    trank = 0

    U, S, V = svd(X[:,:,0], full_matrices = False)
    
    S = np.diag(np.maximum(S-tau,0)) 
    XX[:,:,0] = U@S@V
    tnn  += tnn+np.sum(S)
    logger.info(np.diag(S))

    if n3%2==0:
        halfn3 = int(n3/2) 
    else:
        halfn3 = int(n3/2)+1

    for i in range(1,halfn3):
        U, S, V = svd(X[:,:,i], full_matrices = False)
        S = np.diag(np.maximum(S-tau,0)) 
        XX[:,:,i] = U@S@V
        tnn  += np.sum(S)
        # print(np.diag(S))
        
        XX[:,:,n3-i] = XX[:,:,i].conjugate()

    if n3%2 == 0:
        i = halfn3
        U, S, V = svd(X[:,:,i], full_matrices = False)
        S = np.diag(np.maximum(S-tau,0)) 
        XX[:,:,i] = U@S@V
        tnn  += np.sum(S)
        # print(np.diag(S))

    tnn = tnn/n3
    XX = np.fft.ifft(XX).real
    return XX, tnn

def T_SVD(round, Y, k = 7):
        # if round>25:
        #     k = 6

        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size= Y.shape),torch.empty(size= Y.shape)).cuda()
        X = torch.fft.fft(Y)

        U, S, V = torch.svd(X[:,:,0])
        # print(U,S,V)
        # print("rank_before = {}".format(len(S)))
        S = S.type(torch.complex64)
        if k>=1:
            S = torch.diag(S[0:k])
            XX[:,:,0]  = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)

        if n3%2==0:
            halfn3 = int(n3/2)
        else:
            halfn3 = int(n3/2)+1

        for i in range(1,halfn3):
            U, S, V = torch.svd(X[:,:,i])
            S = S.type(torch.complex64)
            if k>=1:
                S = torch.diag(S[0:k])
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)
            
            XX[:,:,n3-i] = XX[:,:,i].conj()

        if n3%2 == 0:
            i = halfn3
            U, S, V = torch.svd(X[:,:,i])
            S = S.type(torch.complex64)
            if k>=1:
                S = torch.diag(S[0:k])
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)

        XX = torch.fft.ifft(XX).real
        # XX[XX<0]=0
        # print("rank_after = {}".format(k))
        return XX


def ADMM(logger,X):
    '''
    Solve
    min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
    L,E
    by ADMM
    '''
    m, n, l = X.shape
    eps = 1e-6 
    rho = 2
    mu = 1e-3
    mu_max = 1e10 
    max_iters = 100 
    lamb = 0.05
    L = np.zeros((m, n, l), float) 
    E = np.zeros((m, n, l), float) 
    Y = np.zeros((m, n, l), float) 
    iters = 0 
    while True: 
        iters += 1 
        # update L(recovered image) 
        L_new,tnn = SVDShrink2(logger, X - E - (1/mu) * Y, 1/mu) 

        E_new = SoftShrink(X - L_new - (1/mu) * Y, lamb/mu) 

        dY = L_new + E_new - X
        Y += mu * dY
        mu = min(rho * mu, mu_max)
        if converged(L, E, X, L_new, E_new)<eps or iters >= max_iters:
            return L_new, E_new
        else:
            L, E = L_new, E_new
            obj1, obj2 = 0, 0
            for j in range(l):
                obj1 +=np.linalg.norm(E[:,:,j],ord=1)
                obj2 +=np.linalg.norm(dY[:,:,j],ord=2)

            # if iters == 1 or iters%10 == 0:
            logger.info("iters:{}, mu={}, obj={}, err={}, chg={}".format(iters,mu,
            tnn+lamb*obj1,obj2,converged(L, E, X, L_new, E_new)))




