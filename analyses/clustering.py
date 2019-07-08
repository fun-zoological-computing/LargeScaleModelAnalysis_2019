# general modules
import collections
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import numpy as np
from pprint import pprint as pp

# general modules
from ast import literal_eval
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import os
from os.path import join, isfile
from collections import OrderedDict
import string

# analysis modules
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


#### CHANNEL CLUSTERING ####

def normalize_responses(channel_pca_df,num_identifiers=3):
    '''
        Corrects for differences in maximal conductance and inward/outward differences
        for channel by:
            1) normalizing each response, and
            2) multipling inward currents by -1

        Also, performs this for dataframe in place.

    '''

    response_length = len(channel_pca_df.columns.values)-num_identifiers


    for i in channel_pca_df.index.values:

        response = channel_pca_df.iloc[i][0:response_length].values

        max_response = np.max(response)
        norm_response = (response/max_response).tolist()

        for j in range(num_identifiers):
            this_identifier = channel_pca_df.columns.values[-(num_identifiers-j)]
            norm_response.append(channel_pca_df.iloc[i][this_identifier])

        # operates IN PLACE
        channel_pca_df.iloc[i] = norm_response

def compute_protocol_pca(pca_df,protocol_id=None,drop_cols=['Model_ID','Protocol_ID','Channel_Type'],plots=False):

    if protocol_id:
        final_pca_df = pca_df[pca_df['Protocol_ID']==protocol_id]
    else:
        final_pca_df = pca_df



    # drop label columns
    temp_df = final_pca_df.copy()
    if drop_cols:
        df = temp_df.drop(labels=drop_cols,axis='columns')
    else:
        df = temp_df

    # separating out the features
    x = df.loc[:].values

    # choose number of components for pca
    pca = PCA(n_components=.99) # choose n_components s.t. 99% of variance is retained

    # standardize the features
    scaled_x = StandardScaler().fit_transform(x)
    pcs = pca.fit_transform(scaled_x)
    num_pcs = pcs.shape[1] # get the number of components for pca
#     print(num_pcs)

    # redo if PCs (more so for the sake of my code than any technical reason)
    if num_pcs<3:
        # choose number of components for pca
        pca = PCA(n_components=3) # choose n_components s.t. 99% of variance is retained

        # standardize the features
        scaled_x = StandardScaler().fit_transform(x)
        pcs = pca.fit_transform(scaled_x)
        num_pcs = pcs.shape[1] # get the number of components for pca


    scores = pca.score_samples(scaled_x) # Does this need to be the original data?


    # create pca DataFrame
    pc_columns = ['PC '+str(i+1) for i in range(num_pcs)]

    pc_df = pd.DataFrame(data = pcs, columns = pc_columns)


    # toss back in the labels from before - I force all to be type str
    if drop_cols:
        for col_label in drop_cols:
            pc_df[col_label] = pd.Series([str(i) for i in final_pca_df[col_label].values])


    if plots:
        df = pd.DataFrame({'variance':pca.explained_variance_ratio_,
                 'PC':pc_columns})


        # Plotting the raw Explained Variance
        fig = plt.figure(figsize=(12,4))

        # first plot
        sns.barplot(x='PC',y="variance",
                   data=df, color="c")

        plt.ylabel('Total Variance (%)')
        plt.xticks(rotation='vertical')
        plt.title(protocol_id)
        plt.show()


    return pc_df, pca, pcs, scores

#
# def normalize_scores(scores_df,num_identifiers=2):
#     '''
#
#     '''
#
#     vector_length = len(scores_df.columns.values)-num_identifiers
#
#     for  i in scores_df.index.values:
#
#         scores = scores_df.iloc[i][0:vector_length].values
#
#         max_score = np.max(scores)
#         norm_scores = (scores/max_score).tolist()
#
#         for j in range(num_identifiers):
#             this_identifier = scores_df.columns.values[-(num_identifiers-j)]
#             norm_scores.append(scores_df.iloc[i][this_identifier])
#
#         # operates IN PLACE
#         scores_df.iloc[i] = norm_scores




def compute_icg_scores(norm_responses_df,protocols=['ACTIVATION','DEACTIVATION','INACTIVATION'],num_identifiers=3):
    '''
        Include that afterward. Every channel response is reduced to a
        3-dimensional score vector (1 point per protocol or 3 PCs).

    '''


    #### STEP 1) Perform PCA with number PCs to retain 99% explained variance
    pca_df = norm_responses_df.copy() # perform on normalized responses

    pc_dfs = []
    pcas = []
    pcs = []
    score_samples = []

    # compute pca and plot
    for protocol_id in protocols:
        pc_df, pca, pc, scores = compute_protocol_pca(pca_df,protocol_id)

        # collect all
        pc_dfs.append(pc_df)
        pcas.append(pca)
        pcs.append(pc)
        score_samples.append(scores)



    #### Step 2) Normalize condition score vectors (create one score vector per model per protocol)
    final_score_vector = [] # contains all scores = the log-likelihood of samples

    # iterate over protocols
    for i, (score_vector,protocol_id) in enumerate(zip(score_samples,protocols)):

        # normalize the protocol-specific scores by dividing each protocol by std of entries
        condition_scores = score_vector/np.std(score_vector)
        final_score_vector.append(condition_scores) # combined



    #### Step 3) Combine all condition vectors
    final_score_mat = np.array(final_score_vector).T

    # dataframe to store score vectors in
    columns = protocols
    protocol_scores_df = pd.DataFrame(columns=columns)

    for col_i, col in enumerate(columns):
        protocol_scores_df[col] = pd.Series(final_score_mat[:,col_i])




    #### Step 5) Perform PCA again to get final score vector for each model (3 PCs)
    final_pc_df, final_pca, final_pcs, final_scores = compute_protocol_pca(protocol_scores_df,drop_cols=None)


    # template from original response dataframe
    df = norm_responses_df[norm_responses_df['Protocol_ID']=='ACTIVATION']


    channel_models = df.Model_ID.values
    channel_types = df.Channel_Type.values

    # final score vector should be the same length as the number of models
    final_scores_df = final_pc_df.copy()
    final_scores_df['Model_ID'] = pd.Series(channel_models)
    final_scores_df['Channel_Type'] = pd.Series(channel_types)




    return final_scores_df


#### EPHYS CLUSTERING ####



def PCA_and_Cluster(samples_df, raw_samples_df, parent_path = "/", hide_noise = False, remove_noise = False, cluster_all=False,
                    k_means = False, min_cluster_size = 10, kmeans_n_clusters = 6, target_var = .95,
                    cluster_captions=string.ascii_uppercase, axis_captions=['PC0','PC1','PC2'],
                    EDA_plot=True, interactive=False, verbose=True):

    '''
        Perform PCA, hierarchical clustering and silhoette analysis.

        !-- TODO: Move technical notes to a notebook

        Technical notes:
        ----
            - Hierarchical clustering can either be top-down (kmeans) or bottom-up (density-based agglomerative). The bottom-up algorithm used
            is HDBSCAN, an algorithm that uses single-linkage method and produces clusters of varying densities based on cluster stability.
            Single-linkage was preferred as electrophysical properties form a continuum in feature space. It produces dendrograms with
            heterogeneous distance thresholds for each resulting cluster with min_cluster_size as the dominant constraint for cluster cutoffs.
            In addition, this algorithm results in a set of clusters that do not contain the total set of samples, i.e., the resulting clusters
            are not a partition of the feature space! Some samples remain unclustered resulting in a slimmer dendrogram and minimizing false
            positive members of clusters due to noise. (Not sure if this is the best idea. Models are deterministic so lack intrinsic noise terms,
            but there is implicit noise in parameter optimization/hand-tuning.)

            -


        ----
        PARAMETERS:
            - samples_df :
            - parent_path : (str) Forms path from parent clusters to child clusters, ex: /0/2/1/ = cluster 0 -> child cluster 2 -> child cluster 1
            - hide_noise :
            - remove_noise :
            - cluster_all : (MIGHT BE THE SAME AS REMOVE NOISE)
            - k_means :
            - interactive :
            - min_cluster_size :
            - kmeans_n_clusters :
            - target_var :
            - cluster_captions :
            - axis_captions :
            - EDA_plot :
            - verbose :

        OUTPUT:
            - None

    '''

    # Subselect rows based on selected cluster
    df = samples_df[samples_df["ClusterPath"].str.startswith(parent_path)]

    # Normalize feature space
    ss = StandardScaler()

    try:
        x = ss.fit_transform(df.loc[:,prop_names].values)

    # Skip the below if no there are no samples in the cluster path
    except ValueError:
        print('Zero samples were found... skipping clustering')
        print(df) # sanity check

        return None


    # Begin PCA on ephys properties
    x = DataFrame(x,columns=prop_names)

    pca = PCA(svd_solver='full',n_components=target_var)

    principalComponents = pca.fit_transform(x)
    principalDf = DataFrame(data = principalComponents)

    X = principalDf.copy()
    X.index = df.index # Insert model IDs into PCA dataframe

    if verbose:
        print('Dimensions reduced from %s to %s' %(len(prop_names),len(principalDf.columns)))
        print('Number of rows = %s' %X.shape[0]) # sanity check


    # Exploratory cluster analysis of the PCA space - 3D plot, dendrogram, and silhouette analysis
    if EDA_plot:
        plt.figure(figsize=(15, 7))
        plt.axes(projection='3d')
        plt.plot(X[0],X[1], X[2],'bo')
        plt.show()


    # Define number of clusters to explore for silhoette analysis
    range_n_clusters = range(2, 8)

    clusters = []
    widths = []

    for n_clusters in range_n_clusters:
        #clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # compute average
        silhouette_avg = silhouette_score(X, cluster_labels)

        if verbose:
            print("For n_clusters = %s, the average silhoette score is = %s" %(n_clusters, silhouette_avg))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # store cluster sizes and their associated average silhoette score
        clusters.append(n_clusters)
        widths.append(silhouette_avg)



        # !--  NOTE: Isn't used anywhere

        # iterate over each cluster
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)





    if EDA_plot:
        plt.plot(clusters, widths)
        plt.xlabel('Cluster Size')
        plt.ylabel('Average Silhoette Score')
        plt.title('Results of Silhoette Analysis')
        plt.show()








    # Find the properties that are most highly correlated with the first 3 PCA components, i.e., positive loading
    pc_i=0
    comp_names = []
    for pc_i in range(3): # Only the first 3 PCs

        print('Positive vs Negative loadings with PC%s :' %(pc_i+1))

        # compute Pearson's correlation coefficient for pcs and properties
        prop_r = np.array([stats.pearsonr(X[pc_i],df[col])[0] if stats.pearsonr(X[pc_i],df[col])[1] < 0.001 else 0 for col in df.columns[:-3]]) # compute for every column in sample_df
        inds = (-np.abs(prop_r)).argsort() # greatest to least in absolute value


        # show the first five properties aranged in order of the sorted correlations
        print(np.array(prop_names)[inds][:5])
        print(prop_r[inds][:5])

        if EDA_plot:
            fig = plt.figure(figsize=(12,4))
            plt.bar(inds,prop_r[inds])
            plt.set_xlabel(prop_names[inds])
            plt.set_ylabel("Pearson's coefficient, R")
            plt.set_title('Principal Component %s Loading' %pc_i)
            plt.show()

        # show some of these results
        name = ""
        for f in range(3):
            name += ("- R for " if prop_r[inds][f] < 0 else "+ R for ") + prop_names[inds[f]] + '\n'
        comp_names.append(name)
        print(name)
        print("         -----            ")

        #     plt.plot(range(len(pca.components_[0])), pca.components_[i][inds])
        #     plt.show()







    # 3D plot of clusters in PCA space
    X_w_noise = X.copy()
    X_w_noise["Cluster"] = -1
    X_w_noise["WasNoise"] = False

    if remove_noise:
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster.fit_predict(X)
        X = X[cluster.labels_ != -1] # for visualization

    if k_means:
        cluster = KMeans(n_clusters=kmeans_n_clusters,random_state=1)
        cluster.fit_predict(X)

    else:
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster.fit_predict(X)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if hide_noise:
        ax.scatter(
            X[cluster.labels_ != -1][0],
            X[cluster.labels_ != -1][1],
            X[cluster.labels_ != -1][2], depthshade=False,marker='o',
            c=cluster.labels_[cluster.labels_ != -1],
            cmap='rainbow')

    else:
        ax.scatter(
            X[0],
            X[1],
            X[2], depthshade=False,marker='o',
            c=cluster.labels_,
            cmap='rainbow')

    ax.set_xlabel(axis_captions[0])
    ax.set_ylabel(axis_captions[1])
    ax.set_zlabel(axis_captions[2])

    plt.tight_layout()

    centers = []

    if k_means:
        centers = cluster.cluster_centers_
    else:
        labels = np.unique(cluster.labels_) if not hide_noise else np.unique(cluster.labels_[cluster.labels_ != -1])

        for l in labels:
            X_label = X[cluster.labels_ == l]
            center = [np.mean(X_label[c]) for c in range(X.shape[1])]
            centers.append(center)

    pca_centers = centers

    # Show clusters as letters in the plot
    for i, center in enumerate(centers):
        ax.text(center[0],center[1],center[2],cluster_captions[i],size=20)

    plt.show()

    for key in locals().keys():
        globals()[key] = locals()[key]

    import collections
    print(collections.Counter(cluster.labels_))

    # Print cluster summary stats
    for c, center in enumerate(centers):
        dist = np.apply_along_axis(euclidean, 1, X, center)
        dist_sort_is = dist.argsort()
        from pprint import pprint as pp

        pp({"cluster": c,
            "cells": X.iloc[dist_sort_is].index[:5],
            "sd":["{:12.2f}".format(np.std(X.iloc[np.where(cluster.labels_ == c)][pc])) for pc in range(3)],
            "center":["{:12.2f}".format(c) for c in center[0:3]],
           })

    # 3D plot of clusters in RAW feature space
    source_df = raw_samples_df.ix[X.index]

    display_props = ["ISIMedian","AccommodationAtSSMean","AP1DelayMeanStrongStim"]
#     display_props = ["AP1DelayMeanStrongStim","ISIMedian","AccommodationAtSSMean"]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        source_df[display_props[0]],
        source_df[display_props[1]],
        source_df[display_props[2]],
        depthshade=True,
        marker='o',
        c=cluster.labels_,
        cmap='rainbow')

    ax.set_xlabel(display_props[0])
    ax.set_ylabel(display_props[1])
    ax.set_zlabel(display_props[2])

    plt.tight_layout()

    centers = []
    sds = []

    labels = np.unique(cluster.labels_)

    print(display_props)

    for i, l in enumerate(labels):
        X_label = source_df[cluster.labels_ == l]
        center = [np.mean(X_label[prop]) for prop in display_props]
        centers.append(center)

        sd = [np.std(X_label[prop]) for prop in display_props]
        sds.append(sd)

#         ax.text(center[0],center[1],center[2],cluster_captions[i],size=20)

#         print(cluster_captions[i],["{:0.2f}+/-{:0.2f}".format(centers[i][c],sds[i][c]) for c,_ in enumerate(center)])

        reg = smf.ols('AP1DelayMeanStrongStim~ISIMedian',data=X_label).fit()
        print('reg isi v delay params p-s r', reg._results.params, reg._results.pvalues, reg._results.rsquared_adj)

        reg = smf.ols('AP1DelayMeanStrongStim~AccommodationAtSSMean',data=X_label).fit()
        print('reg accom v delay params p-s r', reg._results.params, reg._results.pvalues, reg._results.rsquared_adj)

        print("delay v accom",stats.pearsonr(X_label["AP1DelayMeanStrongStim"],X_label["AccommodationAtSSMean"]))
        print("delay v isi",stats.pearsonr(X_label["AP1DelayMeanStrongStim"],X_label["ISIMedian"]))


    plt.show()



    # Set cluster ids in the transformed DataFrame
    X["Cluster"] = cluster.labels_

    # Set cluster in the DF that also has any noise rows
    for label in X.index:
        X_w_noise.at[label, "Cluster"] = X.at[label, "Cluster"]

    # Assign noise models to the cluster with the closest center
    noise_models = X_w_noise[X_w_noise["Cluster"] == -1].index
    for model in noise_models:
        #find the closest pca space cluster center
        dist = np.apply_along_axis(euclidean, 1, pca_centers, X_w_noise.ix[model][:-2])
        dist_sort_is = dist.argsort()
        X_w_noise.at[model, "Cluster"] = dist_sort_is[0] #[0] stores the closest cluster ID
        X_w_noise.at[model, "WasNoise"] = True

    df["Cluster"] = X_w_noise["Cluster"]
    df["WasNoise"] = X_w_noise["WasNoise"]
    df["ClusterPath"] = parent_path + df["Cluster"].map(str) + "/"


    for label in df.index:
        samples_df.at[label, "ClusterPath"] = df.at[label, "ClusterPath"]
        samples_df.at[label, "Cluster"] = df.at[label, "Cluster"]
        samples_df.at[label, "WasNoise"] = df.at[label, "WasNoise"]

    # Sanity checks
    print('current subset clusters',np.unique(df["ClusterPath"]))
    print('all clusters',np.unique(samples_df["ClusterPath"]))
