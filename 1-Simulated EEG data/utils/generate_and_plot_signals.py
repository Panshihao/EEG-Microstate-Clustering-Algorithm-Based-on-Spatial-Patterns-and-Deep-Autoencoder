import numpy as np
from scipy import signal
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

font = {'family':['Arial', 'Times New Roman'], 'color':'k', 'weight':'normal', 'size':10 }
colors = sns.color_palette('tab10')

def sin(f=0):
    N = 1001
    if f == 0:
        f = random.random() * 40 + 10
    phi = random.random() * N
    y = np.sin(np.linspace(0, f * 2 * np.pi , N) - phi)

    return y

def gauss_pulse(position):
    N = 1001
    t = np.linspace(-0.5, 0.5, N, endpoint=True)

    freq_center = random.random() * 5 + 15 + position * 10
    y_base = sin(f=freq_center)

    random_shift = int(random.random() * 40)

    y, _, _ = signal.gausspulse(t, fc=freq_center, bw=0.35, retquad=True, retenv=True)
    y_shift = np.zeros_like(y)
    y_shift[position*320+random_shift:(position*320+300+random_shift)] = y[350:650]
    y_base += 2*y_shift

    return y_base

def square():
    y = sin()
    y = np.sign(y)
    return y

def tri():
    y = square()
    y = np.cumsum(y)
    y = y/np.max(y)

    return y

def gaussian_white_noise():
    N = 1001
    mu, sigma = 0, 0.1
    y = np.random.normal(mu, sigma, N)
    return y

def chirp():
    N = 1001
    f = 30

    phi = random.random() * N
    t = np.linspace(0, 1, N, endpoint=True)
    y = signal.chirp(t, f0=f, f1=10, t1=1, method='linear', phi=phi)

    return y

def chirp_reverse():
    N = 1001
    f = 30

    phi = random.random() * N
    t = np.linspace(0, 1, N, endpoint=True)
    y = signal.chirp(t, f0=10, f1=f, t1=1, method='linear', phi=phi)

    return y

def generate_signals(stationary=False):
    S = np.empty([6, 1001])

    if stationary:
        S[0,:] = sin()
        S[1,:] = sin()
        S[2,:] = sin()
        S[3,:] = tri()
        S[4,:] = square()
        S[5,:] = gaussian_white_noise()       
    else:
        S[0,:] = gauss_pulse(0)
        S[1,:] = gauss_pulse(1)
        S[2,:] = gauss_pulse(2)
        S[3,:] = chirp()
        S[4,:] = chirp_reverse()
        S[5,:] = gaussian_white_noise()     

    return S


def plot_signals(X, figsize=(2.5, 1.5)):
    num_signals = X.shape[0]
    fig, axlist = plt.subplots(num_signals, 1, sharex=True, sharey=False, figsize=figsize)
    for i, ax in enumerate(axlist):
        ax.plot(X[i], linewidth=0.75, color='k', zorder=10)
        ax.yaxis.set_visible(False)

        if i == 5:
            ax.xaxis.set_minor_locator(MultipleLocator(20))
            ax.set_xticks(np.arange(0,1001,100), 
                ['0','','0.2','','0.4','','0.6','','0.8','','1.0'], fontdict=font);
            ax.set_xlim((0,1000))
        else:
            ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                         labelbottom=False, labelleft=False, direction='out',width=0.4)
        
    plt.xlabel('Time (s)', fontdict=font);

    return fig, axlist


def plot_source_signals(X, scale=1.0, figsize=(2.5,1.5), color='default'):

    fig = plt.figure(figsize=figsize)

    if color == 'default':
        colors = sns.color_palette('tab10')
    else:
        colors = [color] * X.shape[0]

    for i in range(X.shape[0]):
        X_shift = X[i] * scale
        X_shift += i * 5
        
        plt.plot(X_shift, linewidth=0.75, color=colors[i], zorder=10)
        plt.axhline(y=i*5, ls=":",c="gray", linewidth=1)

    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    plt.xlim((0,1000))
    plt.xticks(np.arange(0,1001,100), 
            ['0','','0.2','','0.4','','0.6','','0.8','','1.0'], fontdict=font);
    plt.yticks(np.arange(0,26,5), 
            ['1','2','3','4','5','6'], fontdict=font);

    plt.xlabel('Time (s)', fontdict=font);
    plt.ylabel('Channel', fontdict=font);

    return fig

def find_value_index(arr,min,max):
	pos_min = arr>min
	pos_max =  arr<max
	pos_rst = pos_min & pos_max

	return pos_rst #np.where(pos_rst == True)
