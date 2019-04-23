import numpy as np, os, sys
import data_util
from model.naive_bayes import NaiveBayes
from model.logistic_regression import LogReg
import pickle

def train_save_plot_model(model,save_model_location):
    model.train()
    pickle.dump(model,open(save_model_location,'wb'))
    model.plot_roc()

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    model_type = sys.argv[1]
    data_directory = sys.argv[2]
    save_model_location = sys.argv[3]

    # Find files.
    files = []
    for f in os.listdir(data_directory):
        if os.path.isfile(os.path.join(data_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    subject_data_list = []
    subject_label_list = []
    sepsis_time_label_list = []
    first_n_col = 7
        
    for f in files:
        input_file = os.path.join(data_directory, f)
        individual_data, individual_label = data_util.load_challenge_data_with_label(input_file)
        individual_data = individual_data[:,:first_n_col]
        spesis_time_labels = data_util.get_sepsis_labels(individual_label)
        individual_data = data_util.remove_nans(individual_data)
        if np.size(individual_data) !=0:
            subject_data_list.extend(data_util.add_time_relative_data(individual_data))
            subject_label_list.extend(individual_label)
            sepsis_time_label_list.extend(spesis_time_labels)
    all_data = np.array(subject_data_list)
    all_labels = np.array(subject_label_list)
    all_sepsis_time_labels = np.array(sepsis_time_label_list)
    if model_type == 'NB':
        model = NaiveBayes(all_data, all_labels)
    elif model_type == 'logreg':
        model = LogReg(all_data, all_labels)
    else:
        raise Exception('model name not recognized')
    train_save_plot_model(model,save_model_location)