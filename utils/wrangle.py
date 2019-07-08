# general modules
from ast import literal_eval
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import os
from os.path import join, isfile
from collections import OrderedDict
from scipy.interpolate import interp1d


def confirm_cortical_model(neurolex_term):
    '''
        Used to NeuroLex ID containing the brain area. 'cort' used for neocortex or other similar names.

        ----
        PARAMETERS : ()

        OUTPUT : (bool) Wheter a cortical model was found or not

    '''
    try:
        return 'cort' in neurolex_term
    except TypeError: # if neurolex_term is None, these should be manually validated
        return False

def get_model(nmldb_id,save=False,path_to_save=None):
    '''
        NeuroML-DB API query for model

        TODO: Check if file is store already somewhere and load that instead of calling API.

        ----
        PARAMETER:
            - nmldb_id : (str) Contains NML ID
            - save : (bool)
            - path_to_save : (str)

        OUTPUT:
            - mldb_model_response : (str) Contains all model specifications
    '''


    nmldb_xml_url = 'https://neuroml-db.org/render_xml_file?modelID='

    model_xml_url = nmldb_xml_url + nmldb_id

    xml_name = nmldb_id+'.xml'

    filename = join(path_to_save,xml_name)

    # If file exists, load and return
    if os.path.isfile(filename):
        return None # replace with uploading the model


    model_xml_response = requests.get(model_xml_url)


    # save file
    if save:

        if path_to_save:
            filename = os.path.join(path_to_save,xml_name)
        else:
            filename = xml_name

        with open(filename, 'wb') as file:
            file.write(model_xml_response.content)

    return model_xml_response.text





def get_neuron_model_details(nmldb_id):
    '''
        NeuroML-DB API query for model details

        TODO: Check if file is store already somewhere and load that instead of calling API.

        ----
        PARAMETER:
            - nmldb_id : (str) Contains NML ID

        OUTPUT:
            - mldb_model_response : (str) Contains all model details in JSON format
    '''

    # collects all model details from database
    if nmldb_id in ['all']:
        nmldb_url = 'http://neuroml-db.org/api/models'

        nmldb_model_response = requests.get(nmldb_url)

    # collects single model details
    else:
        nmldb_url = 'http://neuroml-db.org/api/model?id='

        neuron_url = nmldb_url + nmldb_id

        nmldb_model_response = requests.get(neuron_url)

    return nmldb_model_response.json()


def get_channel_model_details(nmldb_id):

    if 'CL' in nmldb_id:
        raise Exception('NMLDB ID corresponds to a cell model!')

    elif nmldb_id in ['all']:
        nmldb_url = 'http://neuroml-db.org/api/models'

        nmldb_model_response = requests.get(nmldb_url)

    else:
        nmldb_url = 'http://neuroml-db.org/api/model?id='

        neuron_url = nmldb_url + nmldb_id

        nmldb_model_response = requests.get(neuron_url)

    return nmldb_model_response.json()

def create_channel_responses_df(ids,desired_var='Current',save_waveforms=False,save_models=False,path_to_models=None):
    '''
        Collects channel waveforms from NeuroML-DB.
            1) query channel details for list of waveform ids.
            2) query api for actual waveform

        TODO: implement model file upload

        ----
        PARAMETER:
            - ids : (list/DataFrame) Contains NML IDs somewhere

        OUTPUT:
            - current_responses_df : (DataFrame) Contains each waveform for every model

    '''

    try: # if dataframe with more than 1 entry, get list of ids
        id_list = ids['Model_ID'].values.tolist()

    except AttributeError: # if dataframe with only 1 entry, get id and put in list
        id_list = [ids['Model_ID']]

    except TypeError: # if list of ids

        if type(ids) == list: # list of IDs case
            id_list = ids

        else: # single ID case - type(ids) == str
            id_list = [ids]

    # instantiate dataframe object
    columns = ['Model_ID','Model_Name','Filename','Channel_Type','Protocol_ID','Start_Time','End_Time','Dt','Response_Type','Response_Values','Response_Times','Clamp_Label','Clamp_Values','Clamp_Times']
    channel_responses_df = pd.DataFrame(columns=columns)


    # iterate through channel ids to grab these
    for nmldb_id in id_list:

        # request models from https://neuroml-db.org/api/
        try:
            channel_model_details = get_channel_model_details(nmldb_id=nmldb_id)
        except ValueError:
            # if no JSON object can be decoded
            print('Model_ID: %s has no decodeable JSON file' %nmldb_id)

            # and skip this model
            continue
        except ConnectionError:
            # if connection was aborted
            print('Model_ID: %s connection aborted' %nmldb_id)

            # and skip this model
            continue

        print('Retrieved model details for NMLDB ID : %s' %nmldb_id)


#        # else
#         channel_model = get_model(nmldb_id=nmldb_id,
#                                         save=save_models,
#                                         path_to_save=path_to_models)


        # get desired details about channel traces
        model_filename = channel_model_details['model']['File_Name']
        model_name = channel_model_details['model']['Name']
        channel_type = channel_model_details['model']['Channel_Type']
        waveform_list = channel_model_details['waveform_list']

        # for current and voltage (these should correspond with one another)
        waveform_ids = [wave['ID'] for wave in waveform_list if wave['Variable_Name']==desired_var]
        start_times = [wave['Time_Start'] for wave in waveform_list if wave['Variable_Name']==desired_var]
        end_times = [wave['Time_End'] for wave in waveform_list if wave['Variable_Name']==desired_var]
        clamp_ids = [wave['ID'] for wave in waveform_list if wave['Variable_Name']=='Voltage']


        this_channel_vars = []
        this_channel_type = []
        this_channel_waveforms = []
        this_channel_names = []
        this_channel_protocols = []
        this_channel_clamp_labels = []
        this_channel_clamp_vals = []
        this_channel_filenames = []
        this_channel_ids = []
        this_channel_var_times = []
        this_channel_starts = []
        this_channel_ends = []
        this_channel_dts = []
        this_channel_clamp_times = []

        # current responses
        for waveform_id, start, end in zip(waveform_ids,start_times,end_times):
            this_waveform = get_model_waveform(waveform_id=waveform_id)

            t, var = this_waveform['Times'], this_waveform['Variable_Values']
            dt = this_waveform['dt_or_atol']

            # convert to numpy.array
            t, var = np.array(literal_eval(t)), np.array(literal_eval(var))

            # dataframe values
            this_channel_waveforms.append(var)
            this_channel_names.append(model_name)
            this_channel_filenames.append(model_filename)
            this_channel_type.append(channel_type)
            this_channel_ids.append(nmldb_id)
            this_channel_vars.append(desired_var)
            this_channel_var_times.append(t)

            this_channel_starts.append(start)
            this_channel_dts.append(dt)
            this_channel_ends.append(end)


        # voltage clamps
        for clamp_id in clamp_ids:
            this_waveform = get_model_waveform(waveform_id=clamp_id)

            protocol, clamp_label = this_waveform['Protocol_ID'],this_waveform['Waveform_Label']
            clamp_t, clamp_var = this_waveform['Times'], this_waveform['Variable_Values']

            # convert to numpy.array
            clamp_t, clamp_var = np.array(literal_eval(clamp_t)), np.array(literal_eval(clamp_var))

            this_channel_protocols.append(protocol)
            this_channel_clamp_labels.append(clamp_label)
            this_channel_clamp_times.append(clamp_t)
            this_channel_clamp_vals.append(clamp_var)


        # update channel traces dataframe
        df = pd.DataFrame(columns=columns)

        df['Model_ID'] = pd.Series(this_channel_ids)
        df['Model_Name'] = pd.Series(this_channel_names)
        df['Filename'] = pd.Series(this_channel_filenames)
        df['Channel_Type'] = pd.Series(this_channel_type)
        df['Protocol_ID'] = pd.Series(this_channel_protocols)
        df['Start_Time'] = pd.Series(this_channel_starts)
        df['End_Time'] = pd.Series(this_channel_ends)
        df['Dt'] = pd.Series(this_channel_dts)
        df['Response_Type'] = pd.Series(this_channel_vars)
        df['Response_Values'] = pd.Series(this_channel_waveforms)
        df['Response_Times'] = pd.Series(this_channel_var_times)
        df['Clamp_Label'] = pd.Series(this_channel_clamp_labels)
        df['Clamp_Values'] = pd.Series(this_channel_clamp_vals)
        df['Clamp_Times'] = pd.Series(this_channel_clamp_times)

        join_frames = [channel_responses_df,df]
        channel_responses_df = pd.concat(join_frames,ignore_index=True)



    return channel_responses_df


def get_model_waveform(waveform_id):

        waveform_url = 'http://neuroml-db.org/api/waveform?id=%s' %waveform_id

        nmldb_waveform_response = requests.get(waveform_url)

        return nmldb_waveform_response.json()


def import_model(nmldb_id,path_to_models=None):

    # first check if model file exists
    filename = nmldb_id + '.xml'
    model_file = os.path.join(path_to_models,filename)
    print(model_file)

    if os.path.isfile(model_file):

        model = ET.parse(model_file)
        return model

    else:
        print('No model with NMLDB ID: %s exists!' %nmldb_id)

def query_model_by_keyword(keyword):

    nmldb_url = 'http://neuroml-db.org/api/search?q='

    keyword_url = nmldb_url + keyword

    nmldb_keyword_response = requests.get(keyword_url)


    return nmldb_keyword_response.json()



def get_list_of_all_cortical_models(tags_df):
    '''
        Takes the complete list of cell model ID in NeuroML-DB to:
            1. search for information about whether cell is a cortical cell model or not,
            2. automatically create a list of those cortical cell models, and
            3. manually add any known missing cortical cell models.

        ----
        PARAMETERS:
            - tags_df : (pandas.DataFrame) Contains all NMLCLXXXXXX IDs

        OUTPUT:
            - cortical_tags_list : (list) Contains updated list of tags with NMLCLXXXXXX IDs

    '''


    models = tags_df['Model_ID'].values
    cortical_tags_list = []

    print('Determining if model is a cortical cell model')
    print('... Showing indeterminant and non-cortical cell models:')
    for nmldb_id in models:

        try:

            this_model_details = get_neuron_model_details(nmldb_id=nmldb_id)

        except:
            print('Requests ConnectionError for ID = %s' %nmldb_id)


        neurolex_details = this_model_details['neurolex_ids']

        # !-- TODO: Put the following into the confirm_cortical_cell
        # !-- TODO: Use these also to figure out if cortical
        keyword_details = this_model_details['keywords']

        model_details = [neurolex_details, keyword_details] # use this as input

        # if two neurolex IDs
        try:
            neurolex_type_1 = neurolex_details[0]['NeuroLex_Term']
            neurolex_type_2 = neurolex_details[1]['NeuroLex_Term']

        # if only one or zero neurolex IDs
        except IndexError:

            try:
                # if flag throws IndexError, then zero neurolex IDs
                neurolex_flag = neurolex_details[0]['NeuroLex_Term']

                # these cells only have one neurolex ID
                neurolex_type_1 = neurolex_details[0]['NeuroLex_Term']
                neurolex_type_2 = None

            except IndexError:

                neurolex_type_1 = None
                neurolex_type_2 = None

        # check if at least one neurolex ID contains infor suggesting that it is cortical
        if confirm_cortical_model(neurolex_type_1) or confirm_cortical_model(neurolex_type_2):

            cortical_tags_list.append(nmldb_id)
        else:

            # !-- NOTE: Manual cases to check for
            # In the case of the Gouwens "visual area" cells, grab these too, there's quite a few
            if 'visual area' in tags_df[tags_df['Model_ID']==nmldb_id]['Name'].iloc[0]:
                cortical_tags_list.append(nmldb_id)

            # Hard-coded cortical model NMLCLXXXXXX IDs missed

            else:
                # Case 1) show which ones don't have any info in Neurolex IDs to manually check OR
                # Case 2) show which ones have no info about being a cortical cell in their Neurolex IDs
                print(tags_df[tags_df['Model_ID']==nmldb_id])



    return cortical_tags_list


def create_clamp_times_df(responses_df):

    model_ids = np.unique(responses_df.Model_ID.values).tolist()
    protocol_ids = np.unique(responses_df.Protocol_ID.values).tolist()

    columns = ['Model_ID','Protocol_ID','Clamp_Times']
    clamp_times_df = pd.DataFrame(columns=columns)

    # get the model-specific step times for each protocol
    for model in model_ids:

        this_model_df = responses_df[responses_df['Model_ID']==model]

        for protocol in protocol_ids:

            all_clamp_times = []

            this_protocol_df = this_model_df[this_model_df['Protocol_ID']==protocol]
            all_sampled_times = this_protocol_df.Clamp_Times.values


            for sampled_times in all_sampled_times:

                desired_times = np.unique([int(i) for i in sampled_times]).tolist()
                all_clamp_times.append(desired_times)

            # grab the largest of these (to avoid cases when the voltage step is 0 => no time point)
            lengths = [len(clamp_times) for clamp_times in all_clamp_times]
            max_length = np.max(lengths)
            desired_idx = np.where((lengths==max_length))[0][0] # first index is fine

            desired_clamp_times = all_clamp_times[desired_idx]


            df = pd.DataFrame(columns=columns)

            df['Model_ID'] = pd.Series(model)
            df['Protocol_ID'] = pd.Series(protocol)
            df['Clamp_Times'] = pd.Series([desired_clamp_times])


            join_frames = [clamp_times_df,df]
            clamp_times_df = pd.concat(join_frames,ignore_index=True)


    return clamp_times_df



def get_sample_intervals_of_interest(responses_df,num_samples=512,interpolate=True,interp_method='linear',reflect_inward=False):
    '''
        Subsamples points in intervals of interest for responses. If interpolation is needed, linear interpolation
        is applied to each timeseries with a resolution equivalent to that given from the subsampling rate.

        NOTE: responses_df should only contain one channel model for now.


        ----
        PARAMETERS:
            - response_df : (pandas.DataFrame) contains responses that need to be subsampled
            - interpolate : (bool) whether to interpolate when data are sparsely stored (default:True)

        OUTPUT:
            - samples : (numpy.array) contains an NxT matrix of subsampled data
                                      (N = # of responses, T = # of sample points)


    '''

    # protocol details (should match dictionary)
    protocol_ids = np.unique(responses_df.Protocol_ID.values).tolist()

    protocol_intervals = {
        'ACTIVATION'   : [1,3],
        'INACTIVATION' : [2,4],
        'DEACTIVATION' : [2,3]
    }

    # check whether an interval is not defined for each protocols (empty list otherwise)
    intervals_defined = [False for key in protocol_ids if key not in protocol_intervals.keys()]

    if not all(intervals_defined):
        raise Exception('%s Protocol_ID(s) have no interval defined!' %len(intervals_defined))


    # instantiate matrix to return
    num_responses = len(responses_df.index.values)
    response_samples = np.zeros((num_responses,num_samples))
    response_times = np.zeros((num_responses,num_samples))

    # get clamp times
    clamp_times_df = create_clamp_times_df(responses_df)

    # for each current trace get the desired subinterval
    for trace_id in range(num_responses):

        protocol_id = responses_df['Protocol_ID'].values[trace_id]
        model_id = responses_df['Model_ID'].values[trace_id]

        # collect interval points
        this_model_df = clamp_times_df[clamp_times_df['Model_ID']==model_id]
        this_model_protocol_df = this_model_df[this_model_df['Protocol_ID']==protocol_id]

        desired_clamp_times = this_model_protocol_df.iloc[0]['Clamp_Times']

        t0, tf = protocol_intervals[protocol_id]
        T0, Tf = desired_clamp_times[t0], desired_clamp_times[tf]

        curr_response = responses_df['Response_Values'].values[trace_id]
        curr_times = responses_df['Response_Times'].values[trace_id]

        # if desired, reflect inward currents
        if reflect_inward and np.min(curr_response)<curr_response[0]: # negative deflection from baseline
            curr_response *= -1

        # ... and shift so that min is zero if not already non-negative
        curr_min = np.min(curr_response)
        if curr_min<0:
            curr_response += np.abs(curr_min)*np.ones_like(curr_response) # shift up

        if curr_min>0:
            curr_response -= np.abs(curr_min)*np.ones_like(curr_response) # shift down


        # interpolate if desired
        if interpolate:

            t_interp = np.linspace(T0,Tf,num_samples)
            dt_interp = t_interp[1]-t_interp[0]


            # linear, 1D interpolation
            response_interp = interp1d(curr_times,curr_response,kind=interp_method)

            # store for return
            response_times[trace_id,:] = t_interp
            response_samples[trace_id,:] = response_interp(t_interp)
        else:
            raise Exception('No other subsampling method implemented - set: interpolate=True')


    return response_times, response_samples
