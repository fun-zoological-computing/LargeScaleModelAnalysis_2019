def plot_tetrahedron_projection(dim_x = 1, dim_y = 2, invert_x=False):
    %matplotlib inline

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
