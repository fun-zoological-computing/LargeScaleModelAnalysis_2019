# general modules
import os
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

# local module
from analyses.clustering import *
from figures.visualization import *

# initial scoring step
load_data = True
compute_scores = True
score_plots = True

# clustering step
load_scores = False
cluster_only = False # must load final_scores_


cwd = os.getcwd()
path2data = os.path.join(cwd,'data')
path2figs = os.path.join(cwd,'figures')





# load model data
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
print('Computing ICGenealogy score vectors ...')

print('    for KCa')
kca_scores_df = compute_icg_scores(norm_responses_df=norm_kca_df)

print('    saving KCa scores')
# save final scores
filename = os.path.join(path2data,'final_scores_kca.csv')
kca_scores_df.to_csv(filename)

raise Exception('BREAK')


################################
#### PLOT CHANNEL SCORES ####
################################
if score_plots:

    ''' Plot 3D individual channel PC scores '''
    dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
    titles = ['Cav','Ih','Kv','Nav','KCa']
    colors = ['r','b','g','y','c']


    for i, df in enumerate(dfs):

        fig, ax = plot_scores(scores_df=df,score_type=titles[i],color=colors[i])

        fig.tight_layout()
        fig.suptitle('Score distributions for different channel types',y=1.02,size='xx-large')
        plt.show()


        filename = 'specific_'+titles[i]+ '_score_clusters'
        path2file = os.path.join(path2figs,filename)
        plt.savefig(path2file,bbox_inches='tight')





    ''' Plot combined 3d channel PC scores '''
    dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
    titles = ['Cav','Ih','Kv','Nav','KCa']
    colors = ['r','b','g','y','c']

    # show 3D scatter for each of the channel types
    fig = plt.figure(figsize=(6,6))

    num_models = 0

    for i, df in enumerate(dfs):

        fig, ax = plot_scores(scores_df=df,score_type=titles[i],
                                color=colors[i],fig=fig)

        samples = df['PC 1']

        plt.legend(titles)

        num_models += len(samples)


    fig.tight_layout()
    fig.suptitle('Score clusters (N=%s) for different channel types' %num_models,y=1.02,size='xx-large')
    plt.show()

    filename = 'all_score_clusters'
    path2file = os.path.join(path2figs,filename)
    plt.savefig(path2file,bbox_inches='tight')





    ''' Plot t-SNE visualization of channel scores '''
    dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df,kca_scores_df]
    titles = ['Cav','Ih','Kv','Nav','KCa']
    colors = ['r','b','g','y','c']

    fig, ax = plot_scores_tsne(scores_dfs=dfs,perplexity=30,multi_perplex=False)

    fig.legend(titles)

    fig.tight_layout()
    plt.show()


    filename = 'channel_score_tsne_embedding'
    path2file = os.path.join(path2figs,filename)
    plt.savefig(path2file,bbox_inches='tight')






#########################################
#### CLUSTER AND PLOT CHANNEL SCORES ####
#########################################
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






dfs = [cav_scores_df,ih_scores_df,kca_scores_df,kv_scores_df,nav_scores_df]
titles = ['Cav','Ih','Kv','Kca','Nav']


for i, df in enumerate(dfs):

    fig = plt.figure(figsize=(16,14),dpi=300)

    temp_df = df.copy()
    models = temp_df.pop('Model_ID')
    model_names = channel_tags_df[channel_tags_df['Model_ID'].isin(models)]['Model_Name'].values.tolist()
    channel = temp_df.pop('Channel_Type')

    # Calculate the euclidean distance between each cluster (linkage_matrix)
    link = 'ward'
    Z = linkage(temp_df,link) # default = 'ward'

    leaves = []

    for nmldb_id, name in zip(models,model_names):
        leaves.append(name + ' ('+ nmldb_id + ')')


    clm = sns.clustermap(temp_df, # inherits from above -- metric='Euclidean',method='ward',
                         row_linkage=Z,
                         linewidths=1,
                         col_cluster=False,
                         yticklabels=leaves)


    plt.setp(clm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

#     plt.tight_layout()
    plt.title(titles[i]+' (N=%s)' %len(models))

    filename = titles[i]+'_clustermap_'+link+'_link'
    path2file = os.path.join(path2figs,filename)
    plt.savefig(path2file,bbox_inches='tight')
