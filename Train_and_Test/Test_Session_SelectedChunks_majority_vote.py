import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, normalize, PowerTransformer
from sklearn.svm import SVC, SVR
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_absolute_error, mean_squared_error, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso
from scipy.stats import zscore
from sklearn.utils.estimator_checks import check_estimator
from elm_kernel import ELM
import math
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.mode.chained_assignment = None  # default='warn'


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


def get_chosen_features(csv_file, data_file, embedding_data, scores_data, percentage, feature_index, ascending_order):
    # Read CSV file
    df_csv = pd.read_csv(csv_file)

    # # Read wav2vec embedding pickle file
    # with open(embedding_file, 'rb') as f:
    #     data = pickle.load(f)
    #     embedding_data = data.to_numpy()

    # # Read pickl file including arousal、valence、dominance
    # with open(scores_file, 'rb') as f:
    #     scores_data = pickle.load(f)

    df_scores = pd.DataFrame(scores_data, columns=['arousal', 'valence', 'dominance'])
    df_scores.set_index(df_csv['clip_path'], inplace=True)

    # choose 'clip_path','speaker_id' and 'Duration' column
    df_combined = pd.DataFrame(df_csv, columns=['clip_path','speaker_id','Duration', 'Label'])
    # df_combined['speaker_id'] = df_csv['speaker_id']
    df_combined['embedding_feature'] = [np.array(embedding) for embedding in embedding_data]
    
    # Merge df_scores and df_combined by clip_path
    df_combined = df_combined.merge(df_scores, left_on='clip_path', right_index=True)
    # df_combined['Absolute_btw_arousal_valence'] = abs(df_combined['arousal'] - df_combined['valence'])
    
    # Merge the dataframes on 'clip_path'
    df_merge = pd.merge(df_combined, data_file[['clip_path']], on='clip_path')
    
    # Removing chunks that duration less than 0.5 second.
    df_merge_gt_1s = df_merge[df_merge['Duration'] >= 0.5]

    # Calculate mean value of 'arousal' and 'valence' column
    mean_arousal_valence = df_merge_gt_1s.groupby('speaker_id')[['arousal', 'valence']].transform('mean')

    # Adding new column 'arousal_diff' and 'valence_diff'
    df_merge_gt_1s['arousal_diff'] = abs(df_merge_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_merge_gt_1s['valence_diff'] = abs(df_merge_gt_1s['valence'] - mean_arousal_valence['valence'])

    # Adding new column 'arousal_valence_diff'
    df_merge_gt_1s['arousal_valence_diff'] = abs(df_merge_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_merge_gt_1s['arousal'] - mean_arousal_valence['arousal'])

    # Sorting data by speaker_id and chosen feature (feature_index)
    df_merge_sorted = df_merge_gt_1s.sort_values(by=['speaker_id', feature_index], ascending=[True, ascending_order])


    # print(df_merge_sorted)
    # df_combined_sorted.to_csv('tmp.txt',index=False, sep='\t')
    # Get top percentage data to form session-level acoustic feature
    if percentage == 100:
        df_top_percentage = df_merge_sorted.groupby('speaker_id').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100)))
    else:
        df_top_percentage = df_merge_sorted.groupby('speaker_id').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100))).drop('speaker_id', axis=1)
    # print(df_top_percentage)
    embedding_np = np.vstack(df_top_percentage['embedding_feature'].to_numpy())
    scores_np = df_top_percentage[['arousal', 'valence', 'dominance']].to_numpy()
    clip_label = pd.DataFrame(df_top_percentage, columns=['Label'])
    
    '''
    # df_top_percentage = df_top_percentage.drop('speaker_id', axis=1, inplace=True)
    # print('speaker_id' in df_top_percentage.columns)
    # Calculating embedding and scores mean value
    avg_embedding = df_top_percentage.groupby('speaker_id')['embedding_feature'].apply(lambda x: np.mean(x.values, axis=0))
    avg_scores = df_top_percentage.groupby('speaker_id')[['arousal', 'valence', 'dominance']].mean()
    # Calculating embedding and scores standard deviation value
    std_embedding = df_top_percentage.groupby('speaker_id')['embedding_feature'].apply(lambda x: np.std(x.values, axis=0, ddof=0))
    ## For Pandas pkg, ddof default value of std() is 0
    std_scores = df_top_percentage.groupby('speaker_id')[['arousal', 'valence', 'dominance']].std(ddof=0)


    ave_embedding_np = np.vstack(avg_embedding.to_numpy())  # to two-dimensional matrix
    avg_scores_np = np.vstack(avg_scores.to_numpy())

    std_embedding_np = np.vstack(std_embedding.to_numpy())  # to two-dimensional matrix
    std_scores_np = np.vstack(std_scores.to_numpy())
    
    clip_label = pd.DataFrame(df_merge_gt_1s, columns=['speaker_id','Label'])
    
    '''

    # return ave_embedding_np, avg_scores_np, std_embedding_np, std_scores_np, clip_label
    return embedding_np, scores_np, clip_label


    
def each_fold_train_dev(mapped_label_file, embedding_data, scores_data, dev_data, train_data, test_data, test_fold, dev_fold, percentage, C_range, kernel, deg):

    Data_use_top_percentage = percentage
    is_classification = True

    feature_index = 'valence_diff'
    ascending_order = False


    dev_embedding, dev_scores, clip_dev_label = get_chosen_features(mapped_label_file, dev_data, embedding_data, scores_data, Data_use_top_percentage, feature_index, ascending_order)
    train_embedding, train_scores, clip_train_label = get_chosen_features(mapped_label_file, train_data, embedding_data, scores_data, Data_use_top_percentage, feature_index, ascending_order)


    X_train = train_embedding
    X_dev = dev_embedding

    # X_train = np.concatenate((train_avg_embedding, train_std_embedding), axis=1)
    # X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding), axis=1)


    # For debug, only using small part of the training and dev dataset
    # X_train = pdem_w2v2_train[:2000, :]
    # X_dev = pdem_w2v2_dev[:200, :]

    # X_train = LIWC_train # WORKS okay without any norm
    # X_dev = LIWC_dev

    # X_train = STATS_train
    # X_dev = STATS_dev


    ### Todo: clean feature and code.

    scaler = StandardScaler().fit(X_train)

    X_train_scale = scaler.transform(X_train)
    X_dev_scale = scaler.transform(X_dev)

    # X_train_scale = np.multiply(np.sign(X_train_scale), np.power(np.abs(X_train_scale), 0.5)) # FOR ELM
    # X_dev_scale = np.multiply(np.sign(X_dev_scale), np.power(np.abs(X_dev_scale), 0.5)) # FOR ELM

    X_train_scale = normalize(X_train_scale)
    X_dev_scale = normalize(X_dev_scale)

    # preparing the label
    # train_label = clip_train_label.drop_duplicates()['Label'].to_numpy()
    # dev_label = clip_dev_label.drop_duplicates()['Label'].to_numpy()
    
    train_label = clip_train_label['Label'].to_numpy()
    dev_label = clip_dev_label['Label'].to_numpy()
    
    

    y_train_sev = train_label
    y_dev_sev = dev_label


    def best_params_elm(C_list, Kernel_list, folds):
        
        val_pred_C_values = []
        
        for C in C_list:
            
            val_pred_kernel_values = []
            for Kernel in Kernel_list:
                    
                elm = ELM(c = C, kernel = Kernel, is_classification = is_classification, weighted = True, deg = deg)
                # why don't use scaled training data?                
                elm.fit(X_train, y_train_sev)
                # elm.fit(X_train_scale, y_train_sev)
                y_pred_dev = elm.predict(X_dev_scale)
                
                if is_classification:
                    y_pred_dev_max = np.argmax(y_pred_dev, axis=1)
                    
                    # y_pred_rounded = [round(pred) for pred in y_pred_dev]
                    y_pred_dev_rounded = y_pred_dev_max
                else:
                    y_pred_dev_rounded = np.round(y_pred_dev)
                
                accuracy = accuracy_score(y_dev_sev, y_pred_dev_rounded)
                
                pred_CCC = concordance_correlation_coefficient(y_dev_sev, y_pred_dev_rounded)
                pred_RMSE = mean_squared_error(y_dev_sev, y_pred_dev_rounded, squared = False)

                # Assuming y_pred_dev and y_dev_sev are binary class labels

                precision = precision_score(y_dev_sev, y_pred_dev_rounded, average='binary', pos_label = 1)
                recall = recall_score(y_dev_sev, y_pred_dev_rounded, average='binary', pos_label = 1)
                f1 = f1_score(y_dev_sev, y_pred_dev_rounded, average='binary', pos_label = 1)

                val_pred_kernel_values.append([pred_CCC, pred_RMSE, accuracy, precision, recall, f1])
                # score_average = np.mean(score_list)
                # RMSE_average = np.mean(score_list_RMSE)
                # UAR_average = np.mean(score_list_UAR)
                
                # val_pred_kernel_values.append([pred_CCC, pred_RMSE])
                
            val_pred_C_values.append(val_pred_kernel_values)
        # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. 
        # If you meant to do this, you must specify 'dtype=object' when creating the ndarray. 
        dev_score_table = np.array(val_pred_C_values, dtype=object)
        
        dev_CCC_score_table = dev_score_table[:,:,0]
        dev_RMSE_score_table = dev_score_table[:,:,1]
        dev_accuracy_score_table = dev_score_table[:,:,2]
        dev_precision_score_table = dev_score_table[:,:,3]
        dev_recall_score_table = dev_score_table[:,:,4]
        dev_f1_score_table = dev_score_table[:,:,5]

        
        ### Optimization Metric ###

        # best_C_idx = np.where(dev_CCC_score_table == np.nanmax(dev_CCC_score_table))[0][0]
        # best_kernel_idx = np.where(dev_CCC_score_table == np.nanmax(dev_CCC_score_table))[1][0]
        
        # best_C_idx = np.where(dev_RMSE_score_table == np.nanmax(dev_RMSE_score_table))[0][0]
        # best_kernel_idx = np.where(dev_RMSE_score_table == np.nanmax(dev_RMSE_score_table))[1][0]
        
        # best_C_idx = np.where(dev_accuracy_score_table == np.nanmax(dev_accuracy_score_table))[0][0]
        # best_kernel_idx = np.where(dev_accuracy_score_table == np.nanmax(dev_accuracy_score_table))[1][0]
        
        # best_C_idx = np.where(dev_precision_score_table == np.nanmax(dev_precision_score_table))[0][0]
        # best_kernel_idx = np.where(dev_precision_score_table == np.nanmax(dev_precision_score_table))[1][0]
        
        # best_C_idx = np.where(dev_recall_score_table == np.nanmax(dev_recall_score_table))[0][0]
        # best_kernel_idx = np.where(dev_recall_score_table == np.nanmax(dev_recall_score_table))[1][0]
        
        best_C_idx = np.where(dev_f1_score_table == np.nanmax(dev_f1_score_table))[0][0]
        best_kernel_idx = np.where(dev_f1_score_table == np.nanmax(dev_f1_score_table))[1][0]
        
                
        dev_best_CCC = np.round_(dev_CCC_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_RMSE = np.round_(dev_RMSE_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_accuracy = np.round_(dev_accuracy_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_precision = np.round_(dev_precision_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_recall = np.round_(dev_recall_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_f1 = np.round_(dev_f1_score_table[best_C_idx, best_kernel_idx], decimals = 4)

        return C_list[best_C_idx], Kernel_list[best_kernel_idx], dev_best_CCC, dev_best_RMSE, dev_best_accuracy, dev_best_precision, dev_best_recall, dev_best_f1
    

    # best_C, best_kernel, dev_best_CCC, dev_best_RMSE, dev_best_accuracy, dev_best_precision, dev_best_recall, dev_best_f1 = best_params_elm(C_list = C_range, 
    #                                                                     Kernel_list = kernels, 
    #                                                                     folds = 5)

    def get_preds(C, Kernel):
        elm = ELM(c = C, kernel = Kernel, is_classification = is_classification, weighted = True, deg = deg)
                                    
        elm.fit(X_train, y_train_sev)
        y_pred_dev = elm.predict(X_dev_scale)
        
        if is_classification:
            y_pred_dev_max = np.argmax(y_pred_dev, axis=1)
            
            # y_pred_rounded = [round(pred) for pred in y_pred_dev]
            y_pred_dev_rounded = y_pred_dev_max
        else:
            y_pred_dev_rounded = np.round(y_pred_dev)
    

        return y_pred_dev_rounded

    dev_pred = get_preds(C_range[0], kernel)
    # verify_summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, dev_summation_pred)
    # verify_summation_pred_RMSE = mean_squared_error(y_dev_sev, dev_summation_pred, squared = False)

    # pred_CCC = np.round_(concordance_correlation_coefficient(y_dev_sev, dev_pred), decimals = 4)
    # pred_MAE = np.round_(mean_absolute_error(y_dev_sev, dev_pred), decimals = 4)
    # pred_RMSE = np.round_(mean_squared_error(y_dev_sev, dev_pred, squared = False), decimals = 4)
    # pred_accuracy = np.round_(accuracy_score(y_dev_sev, dev_pred), decimals = 4)
    # pred_precision = np.round_(precision_score(y_dev_sev, dev_pred, average='weighted'), decimals = 4)
    # pred_recall = np.round_(recall_score(y_dev_sev, dev_pred, average='weighted'), decimals = 4)
    # pred_f1 = np.round_(f1_score(y_dev_sev, dev_pred, average='weighted'), decimals = 4)


    # print('Top data use percentage:', Data_use_top_percentage)
    # print('Best C is:', best_C)
    # print('Is ascending order?', ascending_order)
    # print('Fold number:', dev_fold)
    # print('pred_CCC:', pred_CCC)
    # print('pred_RMSE', pred_RMSE)
    # print('pred_accuracy:', pred_accuracy)
    # print('pred_precision', pred_precision)
    # print('pred_recall:', pred_recall)
    # print('pred_f1', pred_f1)
    

    """    
    df_pred = pd.DataFrame({'Actual PHQ8 Score': y_dev_sev, 'Predicted Score': dev_pred})
    sns.set(font_scale = 1.2)
    g = sns.relplot(data = df_pred, x = 'Actual PHQ8 Score',  y = 'Predicted Score', color = 'b')
    g.ax.axline(xy1=(0, 0), slope=1, dashes=(3, 2))
    g.ax.set(ylim=(-1, 25), xlim = (-1,25))

    # df_dev_pred_symps_regression = pd.DataFrame(dev_symp_pred)
    # df_dev_pred_symps_regression.to_csv('Multi-task regression symptom predictions DEV.csv')

    plt.scatter(y_dev_sev, dev_pred)
    plt.show()
    """
    
    # return best_C, best_kernel, pred_CCC, pred_RMSE, pred_accuracy, pred_precision, pred_recall, pred_f1
    return dev_pred, clip_dev_label
    


def each_fold_train_test(mapped_label_file, embedding_data, scores_data, train_data, test_data, test_fold, percentage, best_C, best_kernel, deg):

    Data_use_top_percentage = percentage
    is_classification = True

    feature_index = 'valence_diff'
    ascending_order = False
    # C_range = [1,2,3,4,5,6,7,8,9,10] # before
    # C_range = [0.01, 0.1, 0.5, 1,2,3,4,5,6,7,8,9,10,20,40,80,100,200,300,600,1024,2048]
    # kernels = ['rbf','linear', 'sigmoid', 'poly']

    train_embedding, train_scores, clip_train_label = get_chosen_features(mapped_label_file, train_data, embedding_data, scores_data, Data_use_top_percentage, feature_index, ascending_order)
    test_embedding, test_scores, clip_test_label = get_chosen_features(mapped_label_file, test_data, embedding_data, scores_data, Data_use_top_percentage, feature_index, ascending_order)

    X_train = train_embedding
    X_test = test_embedding
    # X_train = np.concatenate((train_avg_embedding, train_std_embedding), axis=1)
    # X_test = np.concatenate((test_avg_embedding, test_std_embedding), axis=1)
    
    
    # X_train = np.concatenate((X_dev, X_train), axis = 0)


    # For debug, only using small part of the training and dev dataset
    # X_train = pdem_w2v2_train[:2000, :]
    # X_dev = pdem_w2v2_dev[:200, :]


    ### Todo: clean feature and code.

    scaler = StandardScaler().fit(X_train)

    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # X_train_scale = np.multiply(np.sign(X_train_scale), np.power(np.abs(X_train_scale), 0.5)) # FOR ELM
    # X_dev_scale = np.multiply(np.sign(X_dev_scale), np.power(np.abs(X_dev_scale), 0.5)) # FOR ELM

    X_train_scale = normalize(X_train_scale)
    X_test_scale = normalize(X_test_scale)

    # preparing the label
    train_label = clip_train_label['Label'].to_numpy()
    # dev_label = clip_dev_label.drop_duplicates()['Label'].to_numpy()
    test_label = clip_test_label['Label'].to_numpy()




    y_train_sev = train_label
    y_test_sev = test_label




    def get_preds(C, Kernel):
        elm = ELM(c = C, kernel = Kernel, is_classification = is_classification, weighted = True, deg = deg)
                                    
        elm.fit(X_train, y_train_sev)
        y_pred_test = elm.predict(X_test_scale)
        
        if is_classification:
            y_pred_test_max = np.argmax(y_pred_test, axis=1)
            
            # y_pred_rounded = [round(pred) for pred in y_pred_dev]
            y_pred_test_rounded = y_pred_test_max
        else:
            y_pred_test_rounded = np.round(y_pred_test)
    

        return y_pred_test_rounded

    test_pred = get_preds(best_C, best_kernel)
    # verify_summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, dev_summation_pred)
    # verify_summation_pred_RMSE = mean_squared_error(y_dev_sev, dev_summation_pred, squared = False)
    """
    pred_CCC = np.round_(concordance_correlation_coefficient(y_test_sev, test_pred), decimals = 4)
    pred_MAE = np.round_(mean_absolute_error(y_test_sev, test_pred), decimals = 4)
    pred_RMSE = np.round_(mean_squared_error(y_test_sev, test_pred, squared = False), decimals = 4)
    pred_accuracy = np.round_(accuracy_score(y_test_sev, test_pred), decimals = 4)
    pred_precision = np.round_(precision_score(y_test_sev, test_pred, average='binary', pos_label = 1), decimals = 4)
    pred_recall = np.round_(recall_score(y_test_sev, test_pred, average='binary', pos_label = 1), decimals = 4)
    pred_f1 = np.round_(f1_score(y_test_sev, test_pred, average='binary', pos_label = 1), decimals = 4)

    
    print('Test results using best dev set up!!!')
    print('Top data use percentage:', Data_use_top_percentage)
    print('Best C is:', best_C)
    print('Is ascending order?', ascending_order)
    print('Fold number:', test_fold)
    print('pred_CCC:', pred_CCC)
    print('pred_RMSE', pred_RMSE)
    print('pred_accuracy:', pred_accuracy)
    print('pred_precision', pred_precision)
    print('pred_recall:', pred_recall)
    print('pred_f1', pred_f1)
    


    df_pred = pd.DataFrame({'Actual PHQ8 Score': y_test_sev, 'Predicted Score': test_pred})
    sns.set(font_scale = 1.2)
    g = sns.relplot(data = df_pred, x = 'Actual PHQ8 Score',  y = 'Predicted Score', color = 'b')
    g.ax.axline(xy1=(0, 0), slope=1, dashes=(3, 2))
    g.ax.set(ylim=(-1, 25), xlim = (-1,25))

    # df_dev_pred_symps_regression = pd.DataFrame(dev_symp_pred)
    # df_dev_pred_symps_regression.to_csv('Multi-task regression symptom predictions DEV.csv')

    plt.scatter(y_test_sev, test_pred)
    plt.show()
    """

    # return best_C, best_kernel, pred_CCC, pred_RMSE, pred_accuracy, pred_precision, pred_recall, pred_f1
    return test_pred, clip_test_label
    
 


if __name__ == '__main__':
    

    mapped_label_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/gt0d5_labelled_data_4s.csv'
    # Load validation set fold-lists file
    fold_list_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/interview_fold_list.csv'

    # Load embedding feature and VAD scores file
    PDEM_embedding_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/PDEM_outputs/PDEM/PDEM_4s.pkl'
    VAD_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/PDEM_outputs/VAD/VAD_4s.pkl'
    

    

    # List fold columns
    fold_columns = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    # Data_use_top_percentage = [10,20,30,40,50,60,70,80,90,100]
    Data_use_top_percentage = [30,40,50,60,70,80]

    # Read the CSV file
    df_folds = pd.read_csv(fold_list_file)

    # Read the second CSV file
    df_data = pd.read_csv(mapped_label_file)

    # Load embedding feature and VAD scores file
    # PDEM_embedding_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/PDEM_outputs/PDEM/PDEM.pkl'
    # VAD_file = '/media/legalalien/Data1/Androids-Corpus/data_pre/PDEM_outputs/VAD/VAD.pkl'
    

    # # List fold columns
    # fold_columns = ['fold1', 'fold2', 'fold3', 'fold4']
    # fold_columns_test = ['fold5']
    # # Data_use_top_percentage = [5,10,20,30,40,50,60,70,80,90,100]
    # Data_use_top_percentage = [70]
    
    
    dev_results = pd.DataFrame(columns=['test_fold', 'percentage', 'best_C', 'best_kernel', 'pred_accuracy', 'pred_precision', 'pred_recall', 'pred_f1'])
    test_results = pd.DataFrame(columns=['test_fold', 'percentage', 'test_C', 'test_kernel', 'test_CCC', 'test_RMSE', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1'])

    # C_range = [0.001, 0.01, 0.02, 0.1, 0.5, 1,2,3,4,5,6,7,8,9,10,20,40,100,200,600,1024,2048]
    C_range = [0.01, 0.02, 0.1, 0.5, 1,2,3,4,5,6,7,8,16,32,64,128,256,512]
    # C_range = [64,128,256]
    # C_range = [128]
    # kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    kernels = ['linear']
    # gamma = []
    
    # Read wav2vec embedding pickle file
    with open(PDEM_embedding_file, 'rb') as f:
        data = pickle.load(f)
        embedding_data = data.to_numpy()

    # Read pickl file including arousal、valence、dominance
    with open(VAD_file, 'rb') as f:
        vad_scores_data = pickle.load(f)

    for test_fold in fold_columns:
        test_ids = df_folds[test_fold]
        test_data = df_data[df_data['speaker_id'].isin(test_ids)]
        val_each_params_results = []
        train_data_4_test = df_data[~df_data['speaker_id'].isin(test_ids)]
        
        for C in C_range:
            for percentage in Data_use_top_percentage:
                for kernel in kernels:
                    all_preds = []
                    all_gt = []
                    for val_fold in fold_columns:
                        if val_fold == test_fold:
                            continue

                        validation_ids = df_folds[val_fold]
                        val_data = df_data[df_data['speaker_id'].isin(validation_ids)]
                        train_data = df_data[~df_data['speaker_id'].isin(np.concatenate([validation_ids, test_ids]))]

                        fold_preds, group_label = each_fold_train_dev(mapped_label_file, embedding_data, vad_scores_data, val_data, train_data, test_data, test_fold, val_fold, percentage, [C], kernel, deg = 3)
                        # Append the predictions and ground truth to the lists

                        # Convert fold_preds to a DataFrame
                        fold_preds_df = pd.DataFrame(fold_preds, columns=['prediction'])

                        # Add fold_preds_df as a new column to group_label
                        group_label['prediction'] = fold_preds_df['prediction'].values

                        majority_vote = group_label.groupby('speaker_id')['prediction'].agg(lambda x: x.mode().iloc[0])

                        majority_vote_pred = majority_vote.to_numpy()
                        
                        majority_vote_gt = group_label.groupby('speaker_id')['Label'].agg(lambda x: x.mode().iloc[0])             
                        majority_vote_gt = majority_vote_gt.to_numpy()
                        
                        all_preds.extend(majority_vote_pred)
                        all_gt.extend(majority_vote_gt)

                    # Convert list to numpy array
                    all_preds = np.array(all_preds)
                    all_gt = np.array(all_gt)
                    
                    pred_CCC = np.round_(concordance_correlation_coefficient(all_gt, all_preds), decimals = 4)
                    # pred_MAE = np.round_(mean_absolute_error(all_gt, all_preds), decimals = 4)
                    pred_RMSE = np.round_(mean_squared_error(all_gt, all_preds, squared = False), decimals = 4)
                    pred_f1 = np.round_(f1_score(all_gt, all_preds, average='binary', pos_label = 1), decimals = 4)
                    pred_accuracy = np.round_(accuracy_score(all_gt, all_preds), decimals = 4)
                    pred_precision = np.round_(precision_score(all_gt, all_preds, average='binary', pos_label = 1), decimals = 4)
                    pred_recall = np.round_(recall_score(all_gt, all_preds, average='binary', pos_label = 1), decimals = 4)


                    print('Top data use percentage:', percentage)
                    print('C is:', C)
                    print('kernel is:', kernel)
                    # print('Is ascending order?', ascending_order)
                    print('Fold number:', test_fold)
                    print('pred_CCC:', pred_CCC)
                    print('pred_RMSE', pred_RMSE)
                    print('pred_accuracy:', pred_accuracy)
                    print('pred_precision', pred_precision)
                    print('pred_recall:', pred_recall)
                    print('pred_f1', pred_f1)
                    
                    val_each_params_results.append((pred_f1, pred_accuracy, pred_precision, pred_recall, pred_CCC, pred_RMSE, (percentage, C, kernel)))


        # best_avg_f1, best_avg_accuracy, best_avg_precision, best_avg_recall, best_CCC, best_RMSE, best_params = max(val_each_params_results)
        
        # Find the maximum values (excluding best_params)
        max_values = max(val_each_params_results, key=lambda x: x[:6])

        # Find all tuples with the maximum values (excluding best_params)
        max_tuples = [t for t in val_each_params_results if t[:6] == max_values[:6]]

        # Choose the middle tuple
        middle_tuple = max_tuples[0]

        best_avg_f1, best_avg_accuracy, best_avg_precision, best_avg_recall, best_CCC, best_RMSE, best_params = middle_tuple
        
        # Append a new row to dev_results
        dev_results = dev_results.append({
            'test_fold': test_fold,
            'percentage': best_params[0],
            'best_C': best_params[1],
            'best_kernel': best_params[2],
            'pred_f1': best_avg_f1,
            'pred_accuracy': best_avg_accuracy,
            'pred_precision': best_avg_precision,
            'pred_recall': best_avg_recall,
            'pred_CCC': best_CCC,
            'pred_RMSE': best_RMSE
        }, ignore_index=True)

        
        test_pred, clip_test_label = each_fold_train_test(mapped_label_file, embedding_data, vad_scores_data, train_data_4_test, test_data, test_fold, best_params[0], best_params[1], best_params[2], deg = 3)
       
        # test_pred, clip_test_label = each_fold_train_test(mapped_label_file, embedding_data, vad_scores_data, train_data_4_test, test_data, test_fold, 70, 128, 'linear', deg = 3)
        # Convert fold_preds to a DataFrame
        fold_preds_df = pd.DataFrame(test_pred, columns=['prediction'])

        # Add fold_preds_df as a new column to group_label
        clip_test_label['prediction'] = fold_preds_df['prediction'].values

        majority_vote = clip_test_label.groupby('speaker_id')['prediction'].agg(lambda x: x.mode().iloc[0])

        majority_vote_pred = majority_vote.to_numpy()
        
        majority_vote_gt = clip_test_label.groupby('speaker_id')['Label'].agg(lambda x: x.mode().iloc[0])             
        majority_vote_gt = majority_vote_gt.to_numpy()
        
        pred_CCC = np.round_(concordance_correlation_coefficient(majority_vote_gt, majority_vote_pred), decimals = 4)
        pred_MAE = np.round_(mean_absolute_error(majority_vote_gt, majority_vote_pred), decimals = 4)
        pred_RMSE = np.round_(mean_squared_error(majority_vote_gt, majority_vote_pred, squared = False), decimals = 4)
        pred_accuracy = np.round_(accuracy_score(majority_vote_gt, majority_vote_pred), decimals = 4)
        pred_precision = np.round_(precision_score(majority_vote_gt, majority_vote_pred, average='binary', pos_label = 1), decimals = 4)
        pred_recall = np.round_(recall_score(majority_vote_gt, majority_vote_pred, average='binary', pos_label = 1), decimals = 4)
        pred_f1 = np.round_(f1_score(majority_vote_gt, majority_vote_pred, average='binary', pos_label = 1), decimals = 4)

        
        print('Test results using best dev set up!!!')
        # print('Top data use percentage:', best_params[0])
        # print('Test C is:', best_params[1])
        # print('Is ascending order?', ascending_order)
        print('Fold number:', test_fold)
        print('pred_CCC:', pred_CCC)
        print('pred_RMSE', pred_RMSE)
        print('pred_accuracy:', pred_accuracy)
        print('pred_precision', pred_precision)
        print('pred_recall:', pred_recall)
        print('pred_f1', pred_f1)
        
        
        test_results = test_results.append({
            'test_fold': test_fold,
            'percentage': best_params[0],
            'test_C': best_params[1],
            'test_kernel': best_params[2],
            # 'percentage': 70,
            # 'test_C': 128,
            # 'test_kernel': 'linear',
            'test_f1': pred_f1,
            'test_accuracy': pred_accuracy,
            'test_precision': pred_precision,
            'test_recall': pred_recall,
            'test_CCC': pred_CCC,
            'test_RMSE': pred_RMSE
        }, ignore_index=True)

    # Save the DataFrames to Excel files
    dev_results.to_excel('/media/legalalien/Data1/Androids-Corpus/exp/dev_results_valence_descending_f1_4s_mean_std_majority_vote_large_Crange.xlsx', index=False)
    test_results.to_excel('/media/legalalien/Data1/Androids-Corpus/exp/nested_5CV_valence_descending_f1_4s_mean_std_majority_vote_large_Crange.xlsx', index=False)
    # test_results.to_excel('/media/legalalien/Data1/Androids-Corpus/exp/nested_5CV_valence_descending_f1_best_4all.xlsx', index=False)