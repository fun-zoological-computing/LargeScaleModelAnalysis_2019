# general modules
import os
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import hdbscan


# local module
from analyses.clustering import *
from figures.visualization import *

# initial scoring step
load_data = True
compute_scores = False
score_plots = False

# clustering step
load_scores = True
cluster_only = True # must load final_scores_x_dfs


cwd = os.getcwd()
path2data = os.path.join(cwd,'data')
path2figs = os.path.join(cwd,'figures')



# load minimal model data
if load_data:

    # channel identifiers
    filename = os.path.join(path2data,'channel_tags.csv')
    channel_tags_df = pd.read_csv(filename)

    # normalized current responses
    filename = os.path.join(path2data,'norm_cav_responses.csv')
    norm_cav_df = pd.read_csv(filename,index_col=0)
    norm_cav_df['Channel_Type'] = pd.Series(['Cav' for i in norm_cav_df.index.values])

    filename = os.path.join(path2data,'norm_ih_responses.csv')
    norm_ih_df = pd.read_csv(filename,index_col=0)
    norm_ih_df['Channel_Type'] = pd.Series(['Ih' for i in norm_ih_df.index.values])

    filename = os.path.join(path2data,'norm_kv_responses.csv')
    norm_kv_df = pd.read_csv(filename,index_col=0)
    norm_kv_df['Channel_Type'] = pd.Series(['Kv' for i in norm_kv_df.index.values])

    filename = os.path.join(path2data,'norm_nav_responses.csv')
    norm_nav_df = pd.read_csv(filename,index_col=0)
    norm_nav_df['Channel_Type'] = pd.Series(['Nav' for i in norm_nav_df.index.values])

    filename = os.path.join(path2data,'norm_kca_responses.csv')
    norm_kca_df = pd.read_csv(filename,index_col=0)
    norm_kca_df['Channel_Type'] = pd.Series(['KCa' for i in norm_kca_df.index.values])


################################
#### COMPUTE CHANNEL SCORES ####
################################
if compute_scores:
    print('Computing ICGenealogy score vectors ...')

    print('    for Cav')
    cav_scores_df = compute_icg_scores(norm_responses_df=norm_cav_df)

    print('    saving Cav scores')
    # save final scores
    filename = os.path.join(path2data,'final_scores_cav.csv')
    cav_scores_df.to_csv(filename)



    print('    for Ih')
    ih_scores_df = compute_icg_scores(norm_responses_df=norm_ih_df)

    print('    saving Ih scores')
    # save final scores
    filename = os.path.join(path2data,'final_scores_ih.csv')
    ih_scores_df.to_csv(filename)



    print('    for Kv')
    kv_scores_df = compute_icg_scores(norm_responses_df=norm_kv_df)

    print('    saving Kv scores')
    # save final scores
    filename = os.path.join(path2data,'final_scores_kv.csv')
    kv_scores_df.to_csv(filename)



    rint('    for Nav')
    kca_scores_df = compute_icg_scores(norm_responses_df=norm_nav_df)

    print('    saving Nav scores')
    # save final scores
    filename = os.path.join(path2data,'final_scores_nav.csv')
    nav_scores_df.to_csv(filename)




    print('    for KCa')
    kca_scores_df = compute_icg_scores(norm_responses_df=norm_kca_df)

    print('    saving KCa scores')
    # save final scores
    filename = os.path.join(path2data,'final_scores_kca.csv')
    kca_scores_df.to_csv(filename)


# load everything (Skip PCA)
if load_scores:

    # post-PCA scores
    filename = os.path.join(path2data,'final_scores_cav.csv')
    cav_scores_df = pd.read_csv(filename,index_col=0)

    filename = os.path.join(path2data,'final_scores_ih.csv')
    ih_scores_df = pd.read_csv(filename,index_col=0)

    filename = os.path.join(path2data,'final_scores_kv.csv')
    kv_scores_df = pd.read_csv(filename,index_col=0)

    filename = os.path.join(path2data,'final_scores_nav.csv')
    nav_scores_df = pd.read_csv(filename,index_col=0)

    filename = os.path.join(path2data,'final_scores_kca.csv')
    kca_scores_df = pd.read_csv(filename,index_col=0)


################################
#### PLOT CHANNEL SCORES ####
################################
if score_plots:
    print('Plotting channel score visualizations...')

    ''' Plot 3D individual channel PC scores '''
    dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
    titles = ['Cav','Ih','Kv','Nav','KCa']
    colors = ['r','b','g','y','c']


    for i, df in enumerate(dfs):

        fig, ax = plot_scores(scores_df=df,score_type=titles[i],color=colors[i])


        fig.suptitle('Score distributions for different channel types',y=1.02,size='xx-large')
#         fig.tight_layout()
#         plt.show()


        filename = 'specific_'+titles[i]+ '_score_clusters'
        path2file = os.path.join(path2figs,filename)
        plt.savefig(path2file,bbox_inches='tight')




    ''' Plot t-SNE visualization of channel scores '''
    dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
    titles = ['Cav','Ih','Kv','Nav','KCa']
    colors = ['r','b','g','y','c']

    fig, ax = plot_scores_tsne(scores_dfs=dfs,colors=colors,perplexity=30,multi_perplex=False)

    fig.legend(titles)

#     fig.tight_layout()
#     plt.show()


    filename = 'channel_score_tsne_embedding'
    path2file = os.path.join(path2figs,filename)
    plt.savefig(path2file,bbox_inches='tight')






#########################################
#### CLUSTER AND PLOT CHANNEL SCORES ####
#########################################
print('Performing channel score clustering analysis...')


dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
titles = ['Cav','Ih','Kv','Nav','KCa']


print('   ... running HDBSCAN clusterer to get number of clusters')
hdb_clusterers = []
colors = ['r','b','y','g','c']

all_cluster_sizes = []

for i, df in enumerate(dfs):

    temp_df = df.copy()
    models = temp_df.pop('Model_ID')
    model_names = channel_tags_df[channel_tags_df['Model_ID'].isin(models)]['Model_Name'].tolist()
    channel = temp_df.pop('Channel_Type')

    # assign isolated clusters
    temp_df['Cluster_HDB'] = -1

    leaves = []

    for nmldb_id, name in zip(models,model_names):
        leaves.append(name + ' ('+ nmldb_id + ')')


    #### Perform HDBSCAN clustering ####
    min_cluster_size = 2
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                approx_min_span_tree=True,
                                gen_min_span_tree=True)


    clusterer.fit_predict(temp_df)
    hdb_clusterers.append(clusterer)


    # save new clusters
    df['Cluster_HDB'] = clusterer.labels_

    # get network for plotting
    # tree = clusterer.condensed_tree_
    # tree = clusterer.single_linkage_tree_


    # grab number of clusters for agglomerative clustering
    n_clusters = len(np.unique(df[df['Cluster_HDB']!=-1]['Cluster_HDB'].values))
    print('Number of clusters for %s is = %s' %(titles[i],n_clusters))
    all_cluster_sizes.append(n_clusters)



agg_clusterers = []
agg_dendrograms = []

print('   ... running agglomerative clusterer with those clusters')
for i, df in enumerate(dfs):

    fig = plt.figure(figsize=(16,14),dpi=300)

    temp_df = df.copy()
    models = temp_df.pop('Model_ID')
    model_names = channel_tags_df[channel_tags_df['Model_ID'].isin(models)]['Model_Name'].tolist()
    channel = temp_df.pop('Channel_Type')
    hdb_cluster_labels = temp_df.pop('Cluster_HDB')

    temp_df = temp_df[['PC 1','PC 2','PC 3']]


    ''' Perform Agglomerative Clustering '''
    #### SciPy version ####

    # Calculate the euclidean distance between each cluster (linkage_matrix)
    link = 'ward'
    clusterer = linkage(temp_df,link) # default = 'ward'



    leaves = []

    for nmldb_id, name in zip(models,model_names):
        leaves.append(name + ' ('+ nmldb_id + ')')

    print('       -- plotting')
    # !TODO --- move to visualization .py
    clm = sns.clustermap(temp_df, # inherits from above -- metric='Euclidean',method='ward',
                         row_linkage=clusterer,
                         linewidths=1,
                         col_cluster=False,
                         yticklabels=leaves)


    plt.setp(clm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.title(titles[i]+' (N=%s)' %len(models))

    filename = titles[i]+'_clustermap_'+link+'_link'
    path2file = os.path.join(path2figs,filename)
    plt.savefig(path2file,bbox_inches='tight')



    #### SKLearn version ####
    if all_cluster_sizes[i]<=0:
        print('%s has no clusters' %titles[i])

        # set all to the same cluster
        temp_df['Cluster_AGG'] = 1
        continue
    elif titles[i] in ['Nav']:
        # overwrite this case. Clustering is not conservative enough from HDBSCAN.
        n_clusters = 4 # at least captures resurgent channels
    else:
        n_clusters = all_cluster_sizes[i]

    # assign isolated clusters
    temp_df['Cluster_AGG'] = -1

    # for cluster assignments do this
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')

    clusterer.fit_predict(temp_df)
    agg_clusterers.append(clusterer)

    # save new clusters
    df['Cluster_AGG'] = clusterer.labels_


# create and save cluster dataframes with labels
cav_clusters_df = cav_scores_df[['Model_ID','Cluster_HDB','Cluster_AGG']].copy()
ih_clusters_df = ih_scores_df[['Model_ID','Cluster_HDB','Cluster_AGG']].copy()
kv_clusters_df = kv_scores_df[['Model_ID','Cluster_HDB','Cluster_AGG']].copy()
nav_clusters_df = nav_scores_df[['Model_ID','Cluster_HDB','Cluster_AGG']].copy()
kca_clusters_df = kca_scores_df[['Model_ID','Cluster_HDB','Cluster_AGG']].copy()


clusters_dfs = [cav_clusters_df,ih_clusters_df,kv_clusters_df,nav_clusters_df,kca_clusters_df]

for clusters_df in clusters_dfs:

    names = []
    clusters_df['Model_Name'] = ''

    for i, model_id in enumerate(clusters_df['Model_ID'].values):

        model_name = channel_tags_df[channel_tags_df['Model_ID']==model_id]['Model_Name'].iloc[0]
        names.append(model_name)

    clusters_df['Model_Name'] = names
    # clusters_df['Channel_Type'] = pd.Series(['KCa' for i in clusters_df.index.values])

filename = os.path.join(path2data,'cav_clusters.csv')
cav_clusters_df.to_csv(filename)

filename = os.path.join(path2data,'ih_clusters.csv')
ih_clusters_df.to_csv(filename)

filename = os.path.join(path2data,'kv_clusters.csv')
kv_clusters_df.to_csv(filename)

filename = os.path.join(path2data,'nav_clusters.csv')
nav_clusters_df.to_csv(filename)

filename = os.path.join(path2data,'kca_clusters.csv')
kca_clusters_df.to_csv(filename)
