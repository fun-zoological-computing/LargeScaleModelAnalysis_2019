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
