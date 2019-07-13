import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pprint as pp
from pprint import pprint

from sklearn.manifold import TSNE


def plot_tetrahedron_projection(dim_x = 1, dim_y = 2, invert_x=False):

    ax_labels = ["Median Interspike Interval (ms)",
                 "Mean Accomodation at Steady State (%)",
                 "Delay to 1st AP (ms)"]

    fig = plt.figure()#figsize=(12, 10))
    ax = fig.add_subplot(111)

    ax.scatter(
        source_df[display_props[dim_x]],
        source_df[display_props[dim_y]],
        marker='o',
        c=clusters,
        cmap='rainbow')

    ax.set_xlabel(ax_labels[dim_x])
    ax.set_ylabel(ax_labels[dim_y])

    labels = np.unique(clusters)

    for i, l in enumerate(labels):
        X_label = source_df[clusters == l]
        center = [np.mean(X_label[prop]) for prop in display_props]
        ax.text(center[dim_x],center[dim_y],cluster_captions[i],size=12,backgroundcolor="#2E917A",color="w",bbox={'linewidth':0,'alpha':0.75})


    plt.tight_layout()

    if invert_x:
        plt.gca().invert_xaxis()

    # plt.show()


def plot_pca(pc_df,show_type=False):

    fig, ax = plt.subplots(1,3,figsize=(18, 6))
    ax = ax.flatten()

    which_pcs = [(1,2),(1,3),(2,3)]
    total_pc_channels = len(pc_df.index.values)

    colors = ['r','b','y','g']

    for ax_id, (x_axis,y_axis) in enumerate(which_pcs):
        x_pc = 'PC '+str(x_axis)
        y_pc = 'PC '+str(y_axis)

        ax[ax_id].scatter(pc_df.loc[:, x_pc],
                   pc_df.loc[:, y_pc],
                   c = 'r', s = 50,
                   alpha=0.8,edgecolors='gray')


        ax[ax_id].set_xlabel(x_pc)
        ax[ax_id].set_ylabel(y_pc)

    return fig, ax





def plot_scores(scores_df,score_type,color='b',fig=None):
    # show 3D scatter for each of the channel types
    if fig is None:
        fig = plt.figure(figsize=(7,7))

    ax = fig.add_subplot(111,projection='3d')

    xs = scores_df['PC 1']
    ys = scores_df['PC 2']
    zs = scores_df['PC 3']

    ax.scatter(xs, ys, zs, s=50, c=color, alpha=0.6, edgecolors='gray')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    return fig, ax





def plot_scores_tsne(scores_dfs,clusters_dfs=None,perplexity=30,multi_perplex=False,colors=None):

    # join all final score vectores for t-SNE viz
    orig_columns = scores_dfs[0].columns.values.tolist()
    orig_columns.append('labels')
    all_scores_df = pd.DataFrame(columns=orig_columns)

    channel_types= ['Cav','Ih','Kv','Nav','KCa']

    sizes = [len(df.index.values) for df in scores_dfs]
    total_types = 5
    length = 0
    cluster_labels = []

    for i, (s_df,c_df) in enumerate(zip(scores_dfs,clusters_dfs)):

        s_df['labels'] = c_df['Cluster_AGG'].values+length*np.ones_like(length)

        # create unique cluster ID
        num_clusters = len(np.unique(c_df['Cluster_AGG'].values))
        length+=num_clusters

        cluster_labels+=[channel_types[i]+str(j) for j in range(num_clusters)]


        join_frames = [all_scores_df,s_df]
        all_scores_df = pd.concat(join_frames,ignore_index=True)

    # extract the feature values for each component
    score_data = all_scores_df.values[:,:3].astype('float64')



    # plot multiple perplexities
    if multi_perplex:

        perplexities = [5,30,50,100]

        fig, ax = plt.subplots(2,2,figsize=(8,12),dpi=300)
        ax = ax.ravel()

        for i, perp in enumerate(perplexities):

            # init = 'pca' or 'random'
            tsne = TSNE(n_components=2,
                        init='random',           # default = 'random'
                        random_state=0,
                        perplexity=perp,         # default = 30, should be less than the number of samples
                        n_iter=5000)             # default = 1000

            # t0 = time()
            tsne_projection = tsne.fit_transform(score_data) # can't use transpose
            # tf = time()
            #
            # print('Time elapsed = %s for perplexity = %s' %((tf-t0),perp))

            tsne_shape = tsne_projection.shape


            # plot for each channel_type
            size_ranges = np.array([[0,sizes[0]],
                                    [sizes[0],np.sum(sizes[:2])],
                                    [np.sum(sizes[:2]),np.sum(sizes[:3])],
                                    [np.sum(sizes[:3]),np.sum(sizes[:4])],
                                    [np.sum(sizes[:4]),np.sum(sizes[:5])]])

            ub, _ = np.shape(size_ranges)


            for j in range(ub):
                ax[i].scatter(tsne_projection[size_ranges[j][0]:size_ranges[j][1],0],
                              tsne_projection[size_ranges[j][0]:size_ranges[j][1],1],
                              c=colors[j])

            if i in [2,3]:
                ax[i].set_xlabel('TSNE-1')

            if i in [0,2]:
                ax[i].set_ylabel('TSNE-2')
            ax[i].set_title('Perplexity = %s' %perp)



    else:
        fig = plt.figure(figsize=(7,7),dpi=300)
        ax = fig.add_subplot(111)

        # init = 'pca' or 'random'
        tsne = TSNE(n_components=2,
                    init='random',             # default = 'random'
                    random_state=0,
                    perplexity=perplexity,  # default = 30, should be less than the number of samples
                    n_iter=5000)            # default = 1000

        # t0 = time()
        tsne_projection = tsne.fit_transform(score_data) # can't use transpose
        # tf = time()

        # print('Time elapsed = %s for perplexity = %s' %((tf-t0),perp))

        tsne_shape = tsne_projection.shape
        # print(tsne_shape)

        # start t-SNE reduction
        tsne_columns = ['tSNE-1','tSNE-2']

        # for col in orig_columns[3:]:
        #     tsne_columns.append(col)

        tsne_df = pd.DataFrame(columns=tsne_columns,data=tsne_projection)
        labels = all_scores_df.pop('labels')
        tsne_df['labels'] = labels

        ax.scatter(x='tSNE-1',y='tSNE-2',data=tsne_df,
                   c='labels',cmap='rainbow',s=100,
                   edgecolors='gray',
                   alpha=0.8)


        unique_labels = np.unique(labels)

        for i, l in enumerate(unique_labels):

            X_label =tsne_df[tsne_df['labels'] == l]

            center = [np.mean(X_label[col]) for col in tsne_columns[:2]]
            ax.text(center[0],center[1],cluster_labels[i],size=15, bbox=dict(facecolor='white', alpha=0.4))



        ax.set_xlabel(tsne_columns[0])
        ax.set_ylabel(tsne_columns[1])

    # fig.patch.set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.suptitle('All Channel Score t-SNE Embeddings',y=1.02)

    return fig, ax






def plot_ephys_clusters(samples_df,raw_samples_df,
                        display_props = ['AP1DelayMean','AP2DelayMean','AP2DelayMeanStrongStim'],
                        cluster_path = '/1/1/',
                        cluster_captions=['FS','dRS','B','naRS','RS','aFS'],
                        figsize = (12,10),x_lim=None,y_lim=None,z_lim=None,
                        plot_3d=True,save_fig=False,show_colorbar=False,
                        plot_channels=False,channel_type=None,cond_dens_df=None):
    '''
        Plots clustering results for desired properties. Used in poster.

    '''

    # get desired clusters
    source_df = samples_df[samples_df["ClusterPath"].str.startswith(cluster_path)]
    clusters = source_df["Cluster"]
    source_df = raw_samples_df.loc[source_df.index]

    desired_models = source_df.index.tolist()

    xs = source_df[display_props[0]]
    ys = source_df[display_props[1]]

    if len(display_props)>2:
        zs = source_df[display_props[2]]


    if plot_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if not plot_channels:
            cmap = 'rainbow'
            c = clusters
        else:
            print('Overlaying channel density...')
            cmap = 'viridis'
            reduced_cond_dens_df = cond_dens_df[cond_dens_df['Model_ID'].isin(desired_models)]
            cond_dens_vals = reduced_cond_dens_df[channel_type].values
            c = cond_dens_vals


        sc = ax.scatter(
                xs, ys, zs,
                depthshade=False,
                marker='o',
                c=c,
                edgecolors='gray',
                cmap=cmap,
                s=100,
                alpha=0.8)

        # clean z-axis
        if not z_lim is None:
            ax.set_zlim(z_lim)

        ax.set_zlabel(display_props[2],size='large')


    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)

        if not plot_channels:
            cmap = 'rainbow'
            c = clusters
        else:
            print('Overlaying channel density...')
            cmap = 'viridis'
            reduced_cond_dens_df = cond_dens_df[cond_dens_df['Model_ID'].isin(desired_models)]
            cond_dens_vals = reduced_cond_dens_df[channel_type].values
            c = cond_dens_vals



        sc = ax.scatter(
                xs, ys,
                marker='o',
                c=c,
                edgecolors='gray',
                cmap=cmap,
                s=100,
                alpha=0.8)


    # clean axes
    ax.set_xlabel(display_props[0],size='large')
    ax.set_ylabel(display_props[1],size='large')

    if not x_lim is None:
            ax.set_xlim(x_lim)
    if not y_lim is None:
            ax.set_ylim(y_lim)

    if plot_channels and show_colorbar:

        fig.colorbar(sc,shrink=0.7)
        ax.set_aspect('auto')
        plt.tight_layout()

    else:
        plt.tight_layout()



    # add cluster assignment labels
    labels = np.unique(clusters)

    for i, l in enumerate(labels):
        X_label = source_df[clusters == l]
        center = [np.mean(X_label[prop]) for prop in display_props]
        if plot_3d:
            ax.text(center[0],center[1],center[2],cluster_captions[i],size=20, bbox=dict(facecolor='white', alpha=0.3))
        else:
            ax.text(center[0],center[1],cluster_captions[i],size=20, bbox=dict(facecolor='white', alpha=0.3))




    return fig, ax
