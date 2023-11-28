"""
Script to implement a ‘toy’ example to illustrate how improving temporal variability
at differing time scales could enhance temporal correlations and to plot Figure S1-4

Author: Yawen Shao, created on May 30, 2023
"""

from scipy.fft import ifft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12,
        }

matplotlib.rc('font', **font)

def wavenumber(i, n):
    if i <= 1+n/2:
        kn = i-1
    else:
        kn = i-(n+1)    

    return kn

def covariance_cal(n,L,K,E):
    lambda1 = np.zeros(n)
        
    for i in range(n):
        kn = wavenumber(i+1,n)
        lambda1[i] = np.exp(-0.5*np.power(kn/L,2))
     
    lambda2 = n*lambda1/np.sum(lambda1)
    C = np.real(E@ np.diag(lambda2) @ E.conj().T)
    
    return C

def plot_heatmap(C, n, title):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.7,5))
    label = np.linspace(1,n,n//2,dtype=int)
    
    cmap = plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    levels = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    ### Plot covariance heatmap
    im = ax.imshow(C, cmap=cmap, norm=norm)
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(0,n,2))
    ax.set_yticks(np.arange(0,n,2))
     
    ax.set_xticklabels(label) 
    ax.set_yticklabels(label)
    
    ax.set_title(title, fontsize=14, weight='bold')
            
    cbaxes = fig.add_axes([0.89, 0.1, 0.04, 0.8]) #left, bottom, width, height
    fig.colorbar(im, orientation='vertical', cax=cbaxes,
                    shrink=0.6,
                    ticks=levels
                    )
    
    fig.subplots_adjust(left=0.04,top=0.95,bottom=0.06,right=0.9)
    fig.savefig('./Figures/Periodic_example_heatmap_'+title+'.jpeg', dpi=300)
    
    return

def plot_eigenvector(V, D, n):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(13,11))
    
    for i in range(5):
        ax[i].plot(np.arange(1,n+1), -V[:,2*i])
        ax[i].set_title('Eigenvector/Time scale '+str(2*i+1)+': Variance='+str(np.round(D[2*i],2)), fontsize=17, weight='bold')
        ax[i].set_ylim([-0.3,0.3])
        ax[i].set_xlim([1, n])
        ax[i].set_ylabel('Amplitude', fontsize=14)
    
    ax[4].set_xlabel('Time', fontsize=14)
    
    fig.subplots_adjust(left=0.06,top=0.97,bottom=0.06,right=0.99, hspace=0.4)
    fig.savefig('./Figures/Periodic_example_eigenvector.jpeg', dpi=300)
    
    return

def compare_eigenvalue(D1, D2, n):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.7,5))
    ax.plot(np.arange(1,n+1), D1, label='T=3')
    ax.plot(np.arange(1,n+1), D2, label='T=2')
    
    ax.legend()
    ax.set_title('Eigenvalue spectrums')
    ax.set_xlim([1, n])
    
    ax.set_xlabel('Eigenvector/Time scale', fontsize=14)
    ax.set_ylabel('Eigenvalue/Variance', fontsize=14)

    fig.subplots_adjust(left=0.09,top=0.95,bottom=0.12,right=0.97)
    fig.savefig('./Figures/Periodic_example_eigenvalue_compare.jpeg', dpi=300)
    
    return


if __name__ == '__main__':
    n = 26
    K = 10000
    E = np.sqrt(n)*ifft(np.eye(n))
    # L = n/2
    
    ##### T=3, L = 4/3 * (4.05136/4)*n/18
    L = 4/3 * (4.05136/4)*n/18
    C = covariance_cal(n,L,K,E)
    plot_heatmap(C, n, 'Covariance matrix when T=3')
    V1,D1,H1 = np.linalg.svd(C, full_matrices=False)
    plot_eigenvector(V1, D1, n)
    
    ##### T=2, (4/2)*(4.05136/4)*n/18
    L = 4/2 * (4.05136/4)*n/18
    C = covariance_cal(n,L,K,E)
    plot_heatmap(C, n, 'Covariance matrix when T=2')    
    V2,D2,H2 = np.linalg.svd(C, full_matrices=False)

    ##### Plot eigen spectrum
    compare_eigenvalue(D1, D2, n)
    