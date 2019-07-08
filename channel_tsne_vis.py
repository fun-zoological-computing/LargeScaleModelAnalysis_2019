import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
from time import time
import matplotlib.pyplot as plt

path2save = '/Users/vrhaynes/Desktop/research/data_analysis/meta/NeuroMLDBTools/data'
path2models = os.path.join(path2save,'models')
path2ephys = os.path.join(path2save,'ephys')
path2images = '/Users/vrhaynes/Desktop/research/data_analysis/meta/NeuroMLDBTools/images'




# load everything
filename = os.path.join(path2save,'final_scores_cav.csv')
cav_scores_df = pd.read_csv(filename,index_col=0)

filename = os.path.join(path2save,'final_scores_ih.csv')
ih_scores_df = pd.read_csv(filename,index_col=0)

# filename = os.path.join(path2save,'norm_kca_responses.csv')
# norm_kca_df = pd.read_csv(filename,index_col=0)

filename = os.path.join(path2save,'final_scores_kv.csv')
kv_scores_df = pd.read_csv(filename,index_col=0)

filename = os.path.join(path2save,'final_scores_nav.csv')
nav_scores_df = pd.read_csv(filename,index_col=0)


# get sizes
cav_size = len(cav_scores_df.index.values)
ih_size = len(ih_scores_df.index.values)
kv_size = len(kv_scores_df.index.values)
nav_size = len(nav_scores_df.index.values)


dfs = [cav_scores_df,ih_scores_df,kv_scores_df,nav_scores_df]
titles = ['Cav','Ih','Kv','Nav']
colors = ['r','b','g','y']


# join all final score vectores for t-SNE viz
orig_columns = cav_scores_df.columns.values.tolist()
all_scores_df = pd.DataFrame(columns=orig_columns)

for df in dfs:
    join_frames = [all_scores_df,df]
    all_scores_df = pd.concat(join_frames,ignore_index=True)


# extract the feature values for each component
score_data = all_scores_df.values[:,:3].astype('float64')


''' Start tSNE stuff '''
tsne_columns = ['tSNE-1','tSNE-2']

for col in orig_columns[3:]:
    tsne_columns.append(col)

tsne_df = pd.DataFrame(columns=tsne_columns)

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

    t0 = time()
    tsne_projection = tsne.fit_transform(score_data) # can't use transpose
    tf = time()

    print('Time elapsed = %s for perplexity = %s' %((tf-t0),perp))

    tsne_shape = tsne_projection.shape

    # plot for each channel_type
    sizes = [[0,cav_size],
             [cav_size,cav_size+ih_size],
             [cav_size+ih_size,cav_size+ih_size+kv_size],
             [cav_size+ih_size+kv_size,cav_size+ih_size+kv_size+nav_size]]#,
             # [cav_size+ih_size+kv_size+nav_size,cav_size+ih_size+kv_size+nav_size+kca_size]]


    for j in range(4):
        ax[i].scatter(tsne_projection[sizes[j][0]:sizes[j][1],0],tsne_projection[sizes[j][0]:sizes[j][1],1],c=colors[j])

    if i in [2,3]:
        ax[i].set_xlabel('TSNE-1')

    if i in [0,2]:
        ax[i].set_ylabel('TSNE-2')
    ax[i].set_title('Perplexity = %s' %perp)


    fig.suptitle('Various Channel Response t-SNE Embeddings')

filename = 'channel_tsne_embedding'
path2file = os.path.join(path2images,filename)
plt.savefig(path2file,bbox_inches='tight')


plt.tight_layout()
plt.show()



print('Successful implementation')
