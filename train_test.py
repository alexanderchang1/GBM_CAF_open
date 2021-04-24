
import multiprocessing
import concurrent.futures
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import easygui
import vampireanalysis
import statsmodels.api as sm


def main():

    vampireanalysis.vampire()

    condition_dict, gen_dict = initialize_vamptables()

    expt_export_folder = initialize_cellprofiler_tables()

    joined_dict = match_tables(condition_dict, gen_dict, expt_export_folder)

    zero_strings = ['GBM6', 'GBM43', 'U251']
    one_strings = ['1997T', '2124T']

    x, y = assign_binary(joined_dict, zero_strings, one_strings)

    classifier = train_logit(x, y)

    project_outcomes(joined_dict, classifier)

    logit(x,y)

def logit(x,y):
    x = sm.add_constant(x)

    model = sm.Logit(y, x)

    result = model.fit(method='lbfgs', maxiter=10000)
    print(result.pred_table())
    print(result.summary())
    print(result.summary2())
    results1 = result.summary().as_text()
    results2 = result.summary2().as_text()

    resultFile = open("table1.csv", 'w')
    resultFile.write(results1)
    resultFile.close()

    resultFile = open("table2.csv", 'w')
    resultFile.write(results2)
    resultFile.close()


def project_outcomes(joined_dict, classifier):
    serial_tryp_data = pd.DataFrame(joined_dict['serial_tryp'])
    non_serial_tryp_data = pd.DataFrame(joined_dict['non_serial_tryp'])

    result_serial_tryp = np.array_split(serial_tryp_data, 3)

    for chunk in result_serial_tryp:
        vector_count, total_count, percent_count = apply_classifier(chunk, classifier)
        print(serial_tryp_data)
        print(percent_count)

    result_non_serial_tryp = np.array_split(non_serial_tryp_data, 3)

    for chunk in result_non_serial_tryp:
        vector_count, total_count, percent_count = apply_classifier(chunk, classifier)
        print(serial_tryp_data)
        print(percent_count)


def apply_classifier(data, classifier):
    data.dropna(0, 'any', inplace=True)
    outcome = classifier.predict(data)
    vector_count = outcome
    total_count = len(outcome)
    percent_count = (sum(outcome)/len(outcome))

    return vector_count, total_count, percent_count


def train_logit(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)

    classifier = LogisticRegression(solver='lbfgs', random_state=0, max_iter=10000)

    classifier.fit(x_train, y_train)

    print('Accuracy: {:.2f}'.format(classifier.score(x_test, y_test)))

    return classifier


def assign_binary(joined_dict, zero_strings, one_strings):
    sum_data = pd.DataFrame()

    for condition in joined_dict:

        if str(condition) in (zero_strings or one_strings):

            print(condition)
            current_df = joined_dict[condition]

            if str(condition) in one_strings:
                print("entered " + str(condition))
                current_df['Class'] = 1
            elif str(condition) in zero_strings:
                print("entered GBM6")
                current_df['Class'] = 0

            if sum_data.empty:
                sum_data = current_df
            else:
                sum_data = sum_data.append(current_df)

    sum_data.dropna(0, 'any', inplace=True)

    y = list(sum_data['Class'])
    x = sum_data.loc[:, sum_data.columns != 'Class']

    return x, y


def match_tables(condition_dict, gen_dict, expt_export_folder):
    joined_dict = {}
    for condition in condition_dict:
        print(condition)
        cyt_key = condition_dict[condition][0]
        nuc_key = condition_dict[condition][1]
        cyt_table = gen_dict[cyt_key]
        cyt_table.dropna(0,'any',inplace=True)
        nuc_table = gen_dict[nuc_key]
        nuc_table.dropna(0,'any',inplace=True)
        method = condition_dict[condition][2]

        global expt_cyt_path
        global expt_nuc_path
        global file_cyt_column

        if method == watershed:
            expt_cyt_path = expt_export_folder + r'\MyExpt_Cytoplasm_watershed.csv'
            expt_nuc_path = expt_export_folder + r'\MyExpt_FilteredNuclei_watershed.csv'
            file_cyt_column = 'FileName_Cytoplasm_watershed'

        elif method == propagation:
            expt_cyt_path = expt_export_folder + r'\MyExpt_Cytoplasm_propagation.csv'
            expt_nuc_path = expt_export_folder + r'\MyExpt_FilteredNuclei_propagation.csv'
            file_cyt_column = 'FileName_cytoplasm_propagation'

        global cyt_headers
        global nuc_headers

        cyt_headers = cyt_table.columns
        nuc_headers = nuc_table.columns

        cyt_headers_empty = pd.DataFrame(columns=cyt_headers)
        nuc_headers_empty = pd.DataFrame(columns=nuc_headers)

        del cyt_headers_empty['Filename']
        del cyt_headers_empty['ImageID']
        del cyt_headers_empty['ObjectID']
        del cyt_headers_empty['X']
        del cyt_headers_empty['Y']
        del nuc_headers_empty['Filename']
        del nuc_headers_empty['ImageID']
        del nuc_headers_empty['ObjectID']
        del nuc_headers_empty['X']
        del nuc_headers_empty['Y']
        cyt_suff = cyt_headers_empty.add_suffix('_cytoplasm')
        nuc_suff = nuc_headers_empty.add_suffix('_nucleus')

        joined_table = pd.concat((cyt_suff, nuc_suff), axis=1)

        global joined_table_headers

        joined_table_headers = joined_table.columns



        num_cores = multiprocessing.cpu_count()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(process_row, i, method, nuc_table): i for i in cyt_table.itertuples()}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    joined_table = joined_table.append(future.result())
                except IndexError:
                    print("IndexError occurred.")

        joined_table.to_csv(condition + ".csv")

        joined_dict[condition] = joined_table

    with open('Processed Tables.pickle', 'wb') as handle:
        pickle.dump(joined_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return joined_dict


def initialize_cellprofiler_tables():
    # Prompt user for the path to the folder containing all the cell profiler exports
    expt_export_folder = easygui.diropenbox("Please select the folder containing your CellProfiler Datasheets:")
    return expt_export_folder


def initialize_vamptables():

    # Dataset + prompt user for directory.
    cyto = easygui.fileopenbox("Please select the VAMPIRE directory for cytoplasm images", "cyto_segmented")

    nuc = easygui.fileopenbox("Please select the VAMPIRE directory for nucleus images", "nuc_segmented")

    cyto_df = pd.read_csv(cyto)
    nuc_df = pd.read_csv(nuc)

    print(cyto_df)
    print(nuc_df)

    print(cyto_df['set location'])
    print(nuc_df['set location'])

    gen_dict = {}

    global watershed
    watershed = "watershed"
    global propagation
    propagation = "propagation"

    for condition, path, tag in zip(cyto_df['condition'], cyto_df['set location'], cyto_df['tag']):
        print(condition, path)
        if 'watershed' in tag:
            pathway = path + r'\VAMPIRE datasheet CYT_watershed_segmented.csv'
            method = watershed
        else:
            pathway = path + r'\VAMPIRE datasheet CYT_propagation_segmented.csv'
            method = propagation
        new_df = pd.read_csv(pathway)
        gen_dict[condition] = new_df

    print(gen_dict)

    for condition, path, tag in zip(cyto_df['condition'], cyto_df['set location'], cyto_df['tag']):
        print(condition, path)
        if 'watershed' in tag:
            pathway = path + r'\VAMPIRE datasheet DAPI_watershed_segmented.csv'
            method = watershed
        else:
            pathway = path + r'\VAMPIRE datasheet DAPI_propagation_segmented.csv'
            method = propagation
        new_df = pd.read_csv(pathway)
        gen_dict[condition] = new_df

    print(gen_dict)

    condition_dict = {}

    for condition_cyto, condition_nuc, tag_cyto in zip(cyto_df['condition'], nuc_df['condition'], cyto_df['tag']):
        cond_name = str(condition_cyto).split('_')[0]
        if 'watershed' in tag_cyto:
            condition_dict[cond_name] = [condition_cyto, condition_nuc, watershed]
        else:
            condition_dict[cond_name] = [condition_cyto, condition_nuc, propagation]

    return condition_dict, gen_dict


def process_row(row, method, nuc_table):

    i = row
    filename = i[1]

    expt_cyt_table = pd.read_csv(expt_cyt_path)

    x_coord = i[4]
    y_coord = i[5]

    value = expt_cyt_table.loc[(expt_cyt_table['Location_Center_X'].round() == x_coord)
                               & (expt_cyt_table['Location_Center_Y'].round() == y_coord)
                               & (expt_cyt_table[file_cyt_column] == filename)]

    # print(value)
    df_value = pd.DataFrame(value)
    # print(df_value)
    # print(df_value['Parent_Nuclei'])
    # print(df_value['FileName_CellTrackerGreen'])
    df_value.sort_index(inplace=True)
    parent_nuclei = df_value['Parent_Nuclei']
    base_filename = df_value['FileName_CellTrackerGreen']
    parent_nuclei = parent_nuclei.to_list()
    base_filename = base_filename.to_list()

    expt_nuc_table = pd.read_csv(expt_nuc_path)

    value_nuc = expt_nuc_table.loc[((expt_nuc_table['FileName_CellTrackerGreen'] == base_filename[0])
                                    & (expt_nuc_table['Parent_Nuclei'] == parent_nuclei[0]))]

    expt_nuc_match = value_nuc

    nuc_x_coord = round(expt_nuc_match['Location_Center_X']).squeeze()
    nuc_y_coord = round(expt_nuc_match['Location_Center_Y']).squeeze()

    dapi_filename_prop = expt_nuc_match['FileName_Nuclei_propagation'].squeeze()
    dapi_filename_water = dapi_filename_prop.replace('propagation','watershed')

    if method == watershed:
        dapi_filename = dapi_filename_water
    elif method == propagation:
        dapi_filename = dapi_filename_prop
    else:
        raise Exception("ERROR, METHOD NOT DEFINED OR LOCATED")

    match_nuc = nuc_table.loc[(nuc_x_coord == nuc_table['X']) & (nuc_y_coord == nuc_table['Y'])
                              & (dapi_filename == nuc_table['Filename'])]

    cyt_row = pd.DataFrame(i)
    nuc_row = pd.DataFrame(match_nuc)

    cyt_row = cyt_row.T

    del cyt_row[0]
    cyt_row.reset_index(drop=True, inplace=True)
    nuc_row.reset_index(drop=True, inplace=True)
    cyt_row.columns = cyt_headers
    nuc_row.columns = nuc_headers

    del cyt_row['Filename']
    del cyt_row['ImageID']
    del cyt_row['ObjectID']
    del cyt_row['X']
    del cyt_row['Y']
    del nuc_row['Filename']
    del nuc_row['ImageID']
    del nuc_row['ObjectID']
    del nuc_row['X']
    del nuc_row['Y']
    cyt_suff = cyt_row.add_suffix('_cytoplasm')
    nuc_suff = nuc_row.add_suffix('_nucleus')

    joined_row = pd.concat((cyt_suff, nuc_suff), axis=1, ignore_index=True)

    joined_row.columns = joined_table_headers

    return joined_row


if __name__ == '__main__':
    main()
