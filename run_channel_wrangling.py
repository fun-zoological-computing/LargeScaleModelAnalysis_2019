# general modules
import os
import pandas as pd
import numpy as np


# local module
from utils.wrangle import *

cwd = os.getcwd()
path2save = os.path.join(cwd,'data')


##############################
#### BEGIN DATA WRANGLING ####
##############################

# load if already saved dataframes
filename = os.path.join(path2save,'channel_tags.csv')
channel_tags_df = pd.read_csv(filename)

filename = os.path.join(path2save,'current_responses.pkl')
current_responses_df = pd.read_pickle(filename)





######################################################
#### INTERPOLATE, SUBSAMPLE, APPEND AND NORMALIZE ####
######################################################

# get intervals of interest for each response and interpolate
response_times,response_samples = get_sample_intervals_of_interest(current_responses_df,reflect_inward=True)

# make new dataframe with minimal information
dropped_columns = ['Start_Time','End_Time','Dt','Response_Values','Response_Times','Clamp_Values','Clamp_Times']
interp_responses_df = current_responses_df.drop(labels=dropped_columns,axis='columns')

interp_responses_df['Response_Values'] = pd.Series([response for response in response_samples])
interp_responses_df['Response_Times'] = pd.Series([times for times in response_times])


# make a dataframe with appended responses for dimensionality reduction
columns = interp_responses_df.columns.values
appended_responses_df = pd.DataFrame(columns=columns)

appended_responses_df = appended_responses_df.drop(labels=['Model_Name','Filename','Response_Type','Clamp_Label','Response_Times'],axis='columns')
new_columns = appended_responses_df.columns.values

# Append responses and reduce interpolated responses in dataframe for each unique model
unique_models = np.unique(interp_responses_df['Model_ID'].values)

for model_id in unique_models:

    this_model_df = interp_responses_df[interp_responses_df['Model_ID']==model_id]
    unique_protocols = np.unique(this_model_df['Protocol_ID'].values)

    for protocol_id in unique_protocols:

        this_model_protocol_df = this_model_df[this_model_df['Protocol_ID']==protocol_id]
        channel_type = this_model_protocol_df['Channel_Type'].values

        for i, curr_response in enumerate(this_model_protocol_df.Response_Values.values):

            if i==0:
                total_response = curr_response
            else:
                total_response = np.concatenate((total_response,curr_response),axis=None)



        # Normalize twice since some cases cause non-normalized responses
        max_response = np.max(total_response)
        norm_total_response = (total_response/max_response).tolist()


        df = pd.DataFrame(columns=new_columns)
        df['Model_ID'] = pd.Series(model_id)
        df['Protocol_ID'] = pd.Series(protocol_id)
        df['Channel_Type'] = pd.Series(channel_type)
        df['Response_Values'] = pd.Series([norm_total_response])


        join_frames = [appended_responses_df,df]
        appended_responses_df = pd.concat(join_frames,ignore_index=True)


# save
filename = os.path.join(path2save,'updated_appended_responses.pkl')
appended_responses_df.to_pickle(filename)
