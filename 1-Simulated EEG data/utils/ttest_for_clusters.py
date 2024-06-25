import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import copy
import scipy
import matplotlib
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['font.family']=['Arial', 'Times New Roman']
plt.style.use('default')
mpl.rcParams["axes.unicode_minus"] = False
font = {'family' : ['Arial', 'Times New Roman'], 
    'color'  : 'k',
    'weight' : 'normal',
    'size'   : 10,
    }

def distance_ttest(labels, dist_matric, samp_num=1000, get_distribution=False):
    
    if np.isnan( np.min(dist_matric) ):
        sym_dist_mat = copy.deepcopy(dist_matric)
        for i in range(sym_dist_mat.shape[0]):
            sym_dist_mat[i,i] = 1
            for j in range(0,i):
                sym_dist_mat[i,j] = dist_matric[j,i]
    else:
        sym_dist_mat = dist_matric

    clu_num = int(np.nanmax(labels))
    p_mat = np.full((clu_num+1, clu_num+1), np.nan)

    distance_distribution_dict = {}

    for i in range(1, clu_num+1):
        for j in range(i+1, clu_num+1):

            # sample distance between two different clusters
            index1 = np.argwhere(labels == i).squeeze()
            index2 = np.argwhere(labels == j).squeeze()

            ind1 = np.random.choice(index1, samp_num, replace=True).squeeze()
            ind2 = np.random.choice(index2, samp_num, replace=True).squeeze()
            vals_list = sym_dist_mat[ind1, ind2]

            # null distribution : sample distance within each clusters
            # index = np.hstack((index1, index2))
            ind1 = np.random.choice(index1, samp_num, replace=True).squeeze()
            ind2 = np.random.choice(index1, samp_num, replace=True).squeeze()
            null_vals_list = sym_dist_mat[ind1, ind2]

            ind1 = np.random.choice(index2, samp_num, replace=True).squeeze()
            ind2 = np.random.choice(index2, samp_num, replace=True).squeeze()
            null_vals_list = np.concatenate( (null_vals_list, sym_dist_mat[ind1, ind2]) )

            # save data
            if clu_num == 5:
                distance_distribution_dict[str(i)+str(j)+'null'] = null_vals_list
                distance_distribution_dict[str(i)+str(j)+'samp'] = vals_list
           
            # independent ttest
            pres = scipy.stats.ttest_ind(vals_list, null_vals_list, alternative='greater')
            p_mat[i, j] = pres[1]
            if pres[1] > 0.05:
                print(i, j, pres[1])

    if get_distribution:
        return p_mat, distance_distribution_dict
    else:
        return p_mat


def draw_test_mat(p_mat, corr_p=False, figsize=(1.2, 1.2), ticks=1):
    clu_num = p_mat.shape[0]

    if corr_p:
        p_mat *= ((clu_num-1)*(clu_num-2)*0.5)
    print(np.argwhere(p_mat>0.05))

    font = {'family' : ['Arial', 'Times New Roman'], 'color'  : 'k', 'weight' : 'normal', 'size'   : 10, }
    fig, ax = plt.subplots(figsize=figsize)
    ax0 = ax.matshow(p_mat, cmap='Reds_r', interpolation='none', vmax=0.05);
    clb = fig.colorbar(ax0, fraction=0.045);
    ax.xaxis.set_ticks_position("bottom")
    # plt.grid(color='gray', linestyle=':', linewidth=0.5);
    clb.set_ticks(ticks=[0,0.01,0.02,0.03,0.04,0.05])
    cbar_label = clb.ax.get_xticklabels() + clb.ax.get_yticklabels()
    [lab.set_font('Arial') for lab in cbar_label]  #Times New Roman  #set_fontstyle
    clb.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(1,clu_num,ticks), np.arange(1,clu_num,ticks), fontdict=font);
    ax.set_yticks(np.arange(1,clu_num,ticks), np.arange(1,clu_num,ticks), fontdict=font);

    ax.set_xlim(0.5,clu_num-0.5)
    ax.set_ylim(0.5,clu_num-0.5)
    plt.grid(False)
    for xy in np.arange(0.5, clu_num-0.5, 1):
        plt.axvline(x=xy,c="gray",ls=":",lw=0.5)
        plt.axhline(y=xy,c="gray",ls=":",lw=0.5)

    plt.show();
