import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pprint as pp
from pprint import pprint


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

    plt.show()


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
        ax = fig.add_subplot(1,1,projection='3d')

    xs = df['PC 1']
    ys = df['PC 2']
    zs = df['PC 3']

    ax.scatter(xs, ys, zs, s=50, c=colors[i], alpha=0.6, edgecolors='gray')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_title(score_type+' (N=%s)' %num_models,size='large')

    return fig, ax

def plot_scores_tsne(scores_dfs,perplexity=30,multi_perplex=False):

    # join all final score vectores for t-SNE viz
    orig_columns = cav_scores_df.columns.values.tolist()
    all_scores_df = pd.DataFrame(columns=orig_columns)

    sizes = [len(df.index.values) for df in score_dfs]

    for df in score_dfs:
        join_frames = [all_scores_df,df]
        all_scores_df = pd.concat(join_frames,ignore_index=True)

    # extract the feature values for each component
    score_data = all_scores_df.values[:,:3].astype('float64')


    # start t-SNE reduction
    tsne_columns = ['tSNE-1','tSNE-2']

    for col in orig_columns[3:]:
        tsne_columns.append(col)

    tsne_df = pd.DataFrame(columns=tsne_columns)



    # plot multiple perplexities
    if multi_perplex:

        perplexities = [5,30,50,100]

        fig, ax = plt.subplots(2,2,figsize=(8,12))
        ax = ax.ravel()


        for i, perp in enumerate(perplexities):

            # init = 'pca' or 'random'
            tsne = TSNE(n_components=2,
                        init='random',             # default = 'random'
                        random_state=0,
                        perplexity=perp,         # default = 30, should be less than the number of samples
                        n_iter=5000)            # default = 1000

            # t0 = time()
            tsne_projection = tsne.fit_transform(score_data) # can't use transpose
            # tf = time()
            #
            # print('Time elapsed = %s for perplexity = %s' %((tf-t0),perp))

            tsne_shape = tsne_projection.shape

            # plot for each channel_type
            size_ranges = [[0,cav_size],
                           [cav_size,cav_size+ih_size],
                           [cav_size+ih_size,cav_size+ih_size+kv_size],
                           [cav_size+ih_size+kv_size,cav_size+ih_size+kv_size+nav_size],
                           [cav_size+ih_size+kv_size+nav_size,cav_size+ih_size+kv_size+nav_size+kca_size]]


            for j in range(4):
                ax[i].scatter(tsne_projection[sizes[j][0]:sizes[j][1],0],tsne_projection[sizes[j][0]:sizes[j][1],1],c=colors[j])

            if i in [2,3]:
                ax[i].set_xlabel('TSNE-1')

            if i in [0,2]:
                ax[i].set_ylabel('TSNE-2')
            ax[i].set_title('Perplexity = %s' %perp)





    else:

        fig, ax = plt.subplots(1,1,figsize=(7,7))
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

        # plot for each channel_type
        size_ranges = [[0,sizes[0]],
                       [sizes[0],np.sum(sizes[:1])],
                       [np.sum(sizes[:1]),np.sum(sizes[:2])],
                       [np.sum(sizes[:2]),np.sum(sizes[:3])],
                       [np.sum(sizes[:3]),np.sum(sizes[:4])]]

        ub, _ = np.shape(size_ranges)
        for j in range(ub):
            ax.scatter(tsne_projection[sizes[j][0]:sizes[j][1],0],tsne_projection[sizes[j][0]:sizes[j][1],1],c=colors[j])

        ax.set_xlabel('TSNE-1')
        ax.set_ylabel('TSNE-2')


    fig.suptitle('Various Channel Scores t-SNE Embeddings')

    return fig, ax
