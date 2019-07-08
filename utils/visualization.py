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


# A function for plotting 3-dimensional data
def plot3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data.T)
    minn,maxx = data.min(),data.max()
    ax.set_xlim(minn,maxx)
    ax.set_ylim(minn,maxx)
    ax.set_zlim(minn,maxx)
