import numpy as np, os, sys
import data_util
from model.naive_bayes import NaiveBayes
from model.logistic_regression import LogReg
from model.mog import mog
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


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

    col_idx = list(range(0,7)) + list(range(8,13)) + [15, 18, 23, 25, 28]
    num_cols = len(col_idx)
    cnt_seps = 0
    cnt_total = 0
    for f in files:
        input_file = os.path.join(data_directory, f)
        individual_data, individual_label, col_names = data_util.load_challenge_data_with_label(input_file)
        individual_data = individual_data[:, col_idx]
        # remove NaNs completely
        # data_nan = np.any(np.isnan(individual_data[:, :]), axis=1)
        # individual_data = individual_data[~data_nan, :]
        # individual_label = individual_label[~data_nan]
        spesis_time_labels = data_util.get_sepsis_labels(individual_label)
        # replace NaN with interpolation
        individual_data = data_util.remove_nans(individual_data)
        # keep only sepsis patients
        # if ~np.any(individual_label):
        #     individual_data = []
        if np.size(individual_data) !=0:
            cnt_total += 1
            if np.any(individual_label):
                cnt_seps += 1
            subject_data_list.extend(individual_data)  # removed: data_util.add_time_relative_data(individual_data))
            subject_label_list.extend(individual_label)
            sepsis_time_label_list.extend(spesis_time_labels)
    all_data = np.array(subject_data_list)
    all_labels = np.array(subject_label_list)
    all_sepsis_time_labels = np.array(sepsis_time_label_list)
    cols_used = [col_names[i] for i in col_idx]
    priors = [(cnt_total - cnt_seps)/cnt_total, cnt_seps/cnt_total]
    # all_labels[all_sepsis_time_labels < 12] = 0

    # plot histgrams with kde plot separate charts
    # col_names_new = np.asarray(col_names)[col_idx]
    # for column in range(num_cols):
    #     plt.figure()
    #     title0 = cols_used[column] + " Non-Sepsis (H0)"
    #     title1 = cols_used[column] + " Sepsis (H1)"
    #     sns.distplot(all_data[all_labels == 0, column], hist=True, kde=True,
    #                  kde_kws={'linewidth': 1},
    #                  ax = plt.subplot(1,2,1))
    #     plt.title(title0)
    #     sns.distplot(all_data[all_labels == 1, column], hist=True, kde=True,
    #                  kde_kws={'linewidth': 1},
    #                  ax = plt.subplot(1,2,2))
    #     plt.title(title1)

    # plotting KDE desnities on same chart
    # sns.distplot(all_data[all_labels == 0, column], hist=False, kde=True,
    #              kde_kws={'linewidth': 1},
    #              label='Non Sepsis', )
    # sns.distplot(all_data[all_labels == 1, column], hist=False, kde=True,
    #              kde_kws={'linewidth': 1},
    #              label='Sepsis', )

    #
    if model_type == 'NB':
        model = NaiveBayes(all_data, all_labels, priors)
    elif model_type == 'logreg':
        model = LogReg(all_data, all_labels, priors)
    elif model_type == 'mog':
        model = mog(all_data, all_labels)
    else:
        raise Exception('model name not recognized')
    train_save_plot_model(model, save_model_location)
    costs = [0, 0.05, -1, 0]  # C00(TN), C10(FP), C11(TP), C01(FN) #toptimal
    [Pd, Pf, prob_e, slope] = model.bayesian_risk(cost=costs)
    plt.plot(Pf, Pd, 'r*', label=r'$t_{optimal}$')
    costs_ts = [0, 0.05, -0.25, 1.25]  # C00(TN), C10(FP), C11(TP), C01(FN  #tsepsis
    [Pd_ts, Pf_ts, prob_e_ts, slope_ts] = model.bayesian_risk(cost=costs_ts)
    plt.plot(Pf_ts, Pd_ts, 'b*', label=r'$t_{sepsis}$')
    plt.legend(loc="lower right")



    # # plot generated distributions of each feature of each class overlayed (fig:dist)
    # for column in range(num_cols):
    #     plt.figure()
    #     title = cols_used[column]
    #     sigma = np.sqrt(model.model.sigma_[0, column])
    #     mu = model.model.theta_[0, column]
    #     x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    #     plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Non Sepsis')
    #     sigma = np.sqrt(model.model.sigma_[1, column])
    #     mu = model.model.theta_[1, column]
    #     x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    #     plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Sepsis')
    #     plt.title(title)
    #     plt.legend(loc="lower right")