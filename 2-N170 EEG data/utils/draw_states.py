import mne
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
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

def draw_state_blocks_for_generated_data(state, figsize=(4, 2), colorbar_fraction=0.025, 
                     tmin=0.0, tmax=1.0, colorbar_ticks=[], n_clusters=0, euc_centers=[], rie_centers=[]):    
    
    if n_clusters == 0:
        n_clusters = int(np.nanmax(state))
    
    tp = state.shape[0]
    if len(colorbar_ticks) == 0:
        tick_num = n_clusters
        if n_clusters > 6:
            tick_num = 5
        colorbar_ticks=np.linspace(1, n_clusters, tick_num, endpoint=True).astype(int)
    
    seg = np.full( (n_clusters, tp), np.nan )
    for i, j in enumerate(state):
        seg[:int(j), i] = j

    plt.figure(figsize=figsize)
    pixel = int(100/n_clusters)

    colors = sns.color_palette('Set3') # Set3 pastel2 tab20 
    tab = sns.color_palette('tab20')
    colors.remove(colors[1])
    colors.extend( [tab[1], tab[3], tab[5], tab[7], tab[9], tab[11], tab[13], tab[15] ] )
    colors = colors[:n_clusters]
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors[:n_clusters], N=n_clusters)

    plt.imshow( seg, origin='lower', cmap=my_cmap, interpolation='none', extent=[0, tp, 0, n_clusters*pixel], 
                vmin=1, vmax=n_clusters, aspect='auto');

    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    clb = plt.colorbar(fraction=colorbar_fraction, pad=0.02, ticks=colorbar_ticks); 
    clb.set_ticks(ticks=colorbar_ticks)
    cbar_label = clb.ax.get_xticklabels() + clb.ax.get_yticklabels()
    [lab.set_font('Arial') for lab in cbar_label]  #Times New Roman  #set_fontstyle
    clb.ax.tick_params(labelsize=10)

    clb.set_label(label="State", loc="center" , fontdict=font, labelpad=1) 

    x = np.linspace(0, int(tp), num=9, endpoint=True)
    xticks = np.around( np.linspace(tmin, tmax, num=9, endpoint=True), decimals=1)
    plt.xticks(x, xticks, fontdict=font);
    plt.xlabel('Time (s)', fontdict=font);
        
    ax.yaxis.set_minor_locator(MultipleLocator(pixel))
    if n_clusters<6:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel)
        yticks = np.arange(1, n_clusters+1, 1)
    elif n_clusters<13:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel*2)
        yticks = np.arange(1, n_clusters+1, 2)
    else:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel*3)
        yticks = np.arange(1, n_clusters+1, 3)

    if len(euc_centers) != 0:
        plt.vlines(euc_centers, [0]*len(euc_centers),  np.arange(1,(n_clusters+1))*pixel,
                   colors='k', linestyles='--', linewidth=1.0)
        plt.scatter(euc_centers, (np.arange(1,(n_clusters+1))*pixel)-4, marker='^', 
                    s=10, color='k', alpha=1)
    
    if len(rie_centers) != 0:
        plt.vlines(rie_centers, [0]*len(rie_centers),  np.arange(1,(n_clusters+1))*pixel,
                   colors='k', linestyles=':', linewidth=1.0)
        plt.scatter(rie_centers, (np.arange(1,(n_clusters+1))*pixel)-4, marker='o', 
                    s=10, color='k', alpha=1)        

    plt.yticks(y, yticks, fontdict=font);
    plt.ylabel('State', fontdict=font);

    plt.grid(color='gray', linestyle=':', linewidth=0.5);
    plt.show();    


def draw_state_blocks_for_eeg(state, figsize=(4, 2), colorbar_fraction=0.025, 
                            tmin=0.0, tmax=0.7, colorbar_ticks=[], n_clusters=0, 
                            rie_dist=[], current_cluster=-1):
   
    # 绘图参数
    matplotlib.rcParams['font.family']=['Arial', 'Times New Roman']
    plt.style.use('default')
    mpl.rcParams["axes.unicode_minus"] = False
    font = {'family' : ['Arial', 'Times New Roman'], 
        'color'  : 'k',
        'weight' : 'normal',
        'size'   : 10,
        }
    
    if n_clusters == 0:
        n_clusters = int(np.nanmax(state))
    
    tp = state.shape[0]
    if len(colorbar_ticks) == 0:
        tick_num = n_clusters
        if n_clusters > 6:
            tick_num = 5
        colorbar_ticks=np.linspace(1, n_clusters, tick_num, endpoint=True).astype(int)
    
    seg = np.full( (n_clusters, tp), np.nan )
    for i, j in enumerate(state):
        seg[:int(j), i] = j

    plt.figure(figsize=figsize)
    pixel = int(100/n_clusters)

    colors = sns.color_palette('Set3') # Set3 pastel2 tab20 
    tab = sns.color_palette('tab20')
    colors.remove(colors[1])
    colors.remove(colors[7])
    colors.extend( [tab[1], tab[5], tab[7], tab[9], tab[11], tab[13], tab[17], tab[19]] )
    colors = colors[:n_clusters]
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors[:n_clusters], N=n_clusters)

    plt.imshow( seg, origin='lower', cmap=my_cmap, interpolation='none', extent=[0, tp, 0, n_clusters*pixel], 
                vmin=1, vmax=n_clusters, aspect='auto');

    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(13))

    clb = plt.colorbar(fraction=colorbar_fraction, pad=0.02, ticks=colorbar_ticks); 
    clb.set_ticks(ticks=colorbar_ticks)
    cbar_label = clb.ax.get_xticklabels() + clb.ax.get_yticklabels()
    [lab.set_font('Arial') for lab in cbar_label]  #Times New Roman  #set_fontstyle
    clb.ax.tick_params(labelsize=10)

    clb.set_label(label="State", loc="center" , fontdict=font, labelpad=1) 

    x = np.arange(0, int(tp+5), step=26)
    xticks = np.around( np.linspace(tmin, tmax, num=8, endpoint=True), decimals=1)
    # x = np.linspace(25, 476, num=9, endpoint=True)
    # xticks = np.around( np.linspace(tmin, tmax, num=9, endpoint=True), decimals=1)
    plt.xticks(x, xticks, fontdict=font);
    plt.xlabel('Time (s)', fontdict=font);
        
    # y = np.arange(pixel, (n_clusters*pixel), step=pixel*2)
    # yticks = np.arange(1, n_clusters, 2)
    ax.yaxis.set_minor_locator(MultipleLocator(pixel))
    if n_clusters<6:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel)
        yticks = np.arange(1, n_clusters+1, 1)
    elif n_clusters<13:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel*2)
        yticks = np.arange(1, n_clusters+1, 2)
    else:
        y = np.arange(pixel, ((n_clusters+1)*pixel), step=pixel*3)
        yticks = np.arange(1, n_clusters+1, 3)

    if current_cluster > -1:
        displacement = int(180*current_cluster)

        riemann_dist_symmetry = np.zeros((180, 180))
        for i in range(180):
            for j in range(i+1, 180):
                riemann_dist_symmetry[i,j] = rie_dist[i+displacement, j+displacement]
                riemann_dist_symmetry[j,i] = rie_dist[i+displacement, j+displacement]

        # Riemann distance
        for clu in list(set(state)):
            tmp = riemann_dist_symmetry[state==clu, :]
            tmp2 = tmp[:, state==clu]
            distance = np.sum(tmp2, axis=0)

            index = distance.argmin()
            vline_x = np.argwhere(state==clu)[index][0]

            plt.vlines(vline_x, [0],  clu*pixel,
                    colors='k', linestyles=':', linewidth=1.0)
            plt.scatter(vline_x, (clu*pixel)-4, marker='o', 
                        s=10, color='k', alpha=1)   

    plt.yticks(y, yticks, fontdict=font);
    plt.ylabel('State', fontdict=font);

    plt.grid(color='gray', linestyle=':', linewidth=0.5);
    plt.show();    


def draw_grand_average_topo(epo_list, epochs, n_rows=3, draw_separate=False, cmap='turbo', colorbar=False):
    
    n_clusters = epo_list.shape[0]

    info_tmp = mne.create_info(ch_names=epochs.ch_names, sfreq=1, ch_types='eeg')
    f_evoked = mne.EvokedArray(epo_list.T, info_tmp, tmin=1, nave=None)
    f_evoked.set_montage(epochs.get_montage())

    if not draw_separate:
        f_evoked.plot_topomap(
                times=np.arange(1, n_clusters+1, 1), #n_clusters
                nrows=n_rows,
                time_format="State %1.0f",
                time_unit='s',
                cmap=cmap,
                vlim=(np.percentile(epo_list, 1)*1e6, np.percentile(epo_list, 99)*1e6), #(np.min, np.max),
                show=True,
                colorbar=False,
                size=0.7,
                # res=1000,
                # contours=10,
                # mask_params=dict(markersize=10),
                scalings = dict(eeg=1e+6),
                cbar_fmt='%3.2f',
                units="uV"
            );
        
    else:
        for t in range(1, n_clusters+1):
            f_evoked.plot_topomap(
                    times=t, 
                    nrows=1,
                    time_format="State %1.0f",
                    time_unit='s',
                    cmap=cmap,
                    vlim=(np.min(epo_list[t-1])*1e6, np.max(epo_list[t-1])*1e6),
                    #(np.percentile(epo_list[t-1], 1)*1e6, np.percentile(epo_list[t-1], 99)*1e6),#(np.min, np.max),
                    show=True,
                    colorbar=colorbar,
                    # contours=10,
                    # mask_params=dict(markersize=10),
                    scalings = dict(eeg=1e+6),
                    cbar_fmt='%3.2f',
                    units="uV"
                );
        
    return f_evoked

def draw_topo_diff(epo_diff, epochs, title='State', cmap='turbo', colorbar=False):
    epo_array = np.zeros((3, epo_diff.shape[0]))
    epo_array[0] = epo_diff

    info_tmp = mne.create_info(ch_names=epochs.ch_names, sfreq=1, ch_types='eeg')
    f_evoked = mne.EvokedArray(epo_array.T, info_tmp, tmin=1, nave=None)
    f_evoked.set_montage(epochs.get_montage())

    f_evoked.plot_topomap(
            times=1, 
            nrows=1,
            time_format=title,
            time_unit='s',
            cmap=cmap,
            vlim=(np.min(epo_diff)*1e6, np.max(epo_diff)*1e6),
            show=True,
            colorbar=colorbar,
            # contours=10,
            # mask_params=dict(markersize=10),
            scalings = dict(eeg=1e+6),
            cbar_fmt='%3.2f',
            units="uV"
        );
            
    return f_evoked    