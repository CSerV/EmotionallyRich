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
import warnings
import pandas as pd

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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

with open('SBERT_LIWC_STATS_train.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    SBERT_train = stored_data['SBERT']
    LIWC_train = stored_data['LIWC']
    STATS_train = stored_data['STATS']
    
with open('SBERT_LIWC_STATS_dev.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    SBERT_dev = stored_data['SBERT']
    LIWC_dev = stored_data['LIWC']
    STATS_dev = stored_data['STATS']
    
with open('SBERT_LIWC_STATS_test.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    SBERT_test = stored_data['SBERT']
    LIWC_test = stored_data['LIWC']
    STATS_test = stored_data['STATS']

# Load session-level pdem wav2vec embedding

# Load map file
train_mapped_label_file = '/media/legalalien/Data2/AVEC2019_DAIC_WOZ/split_audio_labels/detailed_train.csv'
dev_mapped_label_file = '/media/legalalien/Data2/AVEC2019_DAIC_WOZ/split_audio_labels/detailed_dev.csv'


# Load feature file
PDEM_w2v2_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_train/w2v2_train.pkl'
PDEM_w2v2_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_dev/w2v2_dev.pkl'
VAD_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_train/vad_train.pkl'
VAD_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_dev/vad_dev.pkl'



def get_chosen_features(csv_file, embedding_file, scores_file, percentage, feature_index, ascending_order):
    # Read CSV file
    df_csv = pd.read_csv(csv_file)

    # Read wav2vec embedding pickle file
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
        embedding_data = data.to_numpy()

    # Read pickl file including arousal、valence、dominance
    with open(scores_file, 'rb') as f:
        scores_data = pickle.load(f)

    # Convert scores_data to DataFrame and set the ID column as the index
    df_scores = pd.DataFrame(scores_data, columns=['arousal', 'valence', 'dominance'])
    df_scores.set_index(df_csv['Wav-path'], inplace=True)

    # Merge the two DataFrames
    df_combined = pd.DataFrame(df_csv, columns=['Wav-path','AVECParticipant_ID','Duration'])
    # df_combined['AVECParticipant_ID'] = df_csv['AVECParticipant_ID']
    df_combined['embedding_feature'] = [np.array(embedding) for embedding in embedding_data]
    
    # Merge the two DataFrames
    df_combined = df_combined.merge(df_scores, left_on='Wav-path', right_index=True)
    # df_combined['Absolute_btw_arousal_valence'] = abs(df_combined['arousal'] - df_combined['valence'])

    # Removing chunks that duration less than 1 second.
    df_combined_gt_1s = df_combined[df_combined['Duration'] >= 0.5]

    # Calculate the mean of arousal, valence and dominance for each participant
    mean_arousal_valence = df_combined_gt_1s.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].transform('mean')
    # mean_arousal_valence = df_combined_gt_1s.groupby('AVECParticipant_ID')[['arousal', 'valence']].transform('mean')

    df_combined_gt_1s.loc[:, 'arousal_diff'] = abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s.loc[:, 'valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence'])
    df_combined_gt_1s.loc[:, 'dominance_diff'] = abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])
    df_combined_gt_1s.loc[:, 'arousal_valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s.loc[:, 'arousal_dominance_diff'] = abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s.loc[:, 'valence_dominance_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])
    df_combined_gt_1s.loc[:, 'arousal_valence_dominance_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal']) + abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])

    # df_combined_gt_1s['arousal_diff'] = abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    # df_combined_gt_1s['valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence'])

    # adding new column 'arousal_valence_diff'
    # df_combined_gt_1s['arousal_valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])

    # Sort the DataFrame by 'AVECParticipant_ID' and 'Absolute_btw_arousal_valence'
    df_combined_sorted = df_combined_gt_1s.sort_values(by=['AVECParticipant_ID', feature_index], ascending=[True, ascending_order])

    # df_combined_sorted = df_combined.sort_values(by=['AVECParticipant_ID'], ascending=[True])
    # print(df_combined_sorted)
    # df_combined_sorted.to_csv('tmp.txt',index=False, sep='\t')
    # Group by 'AVECParticipant_ID' and get the top percentage of the ranked DataFrame
    if percentage == 100:
        df_top_percentage = df_combined_sorted.groupby('AVECParticipant_ID').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100)))
    else:
        df_top_percentage = df_combined_sorted.groupby('AVECParticipant_ID').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100))).drop('AVECParticipant_ID', axis=1)
    # df_top_percentage = df_top_percentage.drop('AVECParticipant_ID', axis=1, inplace=True)
    
    # Get session-level embedding by averaging the embeddings
    avg_embedding = df_top_percentage.groupby('AVECParticipant_ID')['embedding_feature'].apply(lambda x: np.mean(x.values, axis=0))
    avg_scores = df_top_percentage.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].mean()
    std_embedding = df_top_percentage.groupby('AVECParticipant_ID')['embedding_feature'].apply(lambda x: np.std(x.values, axis=0, ddof=0))
    std_scores = df_top_percentage.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].std(ddof=0)
    # Convert to numpy array
    ave_embedding_np = np.vstack(avg_embedding.to_numpy())  # to two-dimensional matrix
    avg_scores_np = np.vstack(avg_scores.to_numpy())
    std_embedding_np = np.vstack(std_embedding.to_numpy())  # to two-dimensional matrix
    std_scores_np = np.vstack(std_scores.to_numpy())

    return ave_embedding_np, avg_scores_np, std_embedding_np, std_scores_np


# To-do
# 1. Not just get mean value for the chosen chunk. Also std.
# 2. The mean value calculation of VAD is possiblely not right. I need check and discuss with Yun.
# Because the results was not consistent with using all features.
Data_use_top_percentage_range = [10,20,30,40,50,60,70,80,90,100]
result_dict = {'Data_use_top_percentage': [], 'RMSE': [], 'CCC': []}

for Data_use_top_percentage in Data_use_top_percentage_range:
    # Data_use_top_percentage = 50

    feature_index = 'valence_diff'
    ascending_order = True
    C_range = [1,2,3,4,5,6,7,8,9,10] # before
    kernels = ['rbf','linear', 'sigmoid', 'poly']

    dev_avg_embedding, dev_avg_scores, dev_std_embedding, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
    train_avg_embedding, train_avg_scores, train_std_embedding, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)

    ## To do: concatenate features selected by arousal and valence
    # feature_index = 'arousal_diff'
    # dev_avg_embedding_as, dev_avg_scores, dev_std_embedding_as, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
    # train_avg_embedding_as, train_avg_scores, train_std_embedding_as, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)

    # feature_index = 'valence_diff'
    # dev_avg_embedding_vl, dev_avg_scores, dev_std_embedding_vl, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
    # train_avg_embedding_vl, train_avg_scores, train_std_embedding_vl, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)

    # X_train = np.concatenate((train_avg_embedding_as, train_avg_embedding_vl), axis=1)
    # X_dev = np.concatenate((dev_avg_embedding_as, dev_avg_embedding_vl), axis=1)

    # X_train = pdem_w2v2_trai.to_numpy() # Works well!
    # X_dev = pdem_w2v2_dev.to_numpy()

    # X_train = vad_w2v2_train
    # X_dev = vad_w2v2_dev

    # X_train = train_avg_embedding
    # X_dev = dev_avg_embedding

    ## The following feature combination (PDEM mean + PDEM std + VAD std + SBERT) got the best performance!
    # X_train = train_avg_embedding
    # X_dev = dev_avg_embedding

    X_train = np.concatenate((train_avg_embedding, train_std_embedding), axis=1)
    X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding), axis=1)

    # X_train = np.insert(X_train, 0, 1, axis=1)
    # X_dev = np.insert(X_dev, 0, 1, axis=1)

    # X_train = np.concatenate((train_avg_embedding, train_std_embedding, SBERT_train), axis=1)
    # X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding, SBERT_dev), axis=1)


    # For debug, only using small part of the training and dev dataset
    # X_train = pdem_w2v2_train[:2000, :]
    # X_dev = pdem_w2v2_dev[:200, :]

    # X_train = LIWC_train # WORKS okay without any norm
    # X_dev = LIWC_dev

    # X_train = STATS_train
    # X_dev = STATS_dev

    # X_train = np.concatenate((SBERT_train, LIWC_train), axis = 1)
    # X_dev = np.concatenate((SBERT_dev, LIWC_dev), axis = 1)

    # X_train = np.concatenate((SBERT_train, STATS_train), axis = 1)
    # X_dev = np.concatenate((SBERT_dev, STATS_dev), axis = 1)

    # X_train = np.concatenate((STATS_train, LIWC_train), axis = 1)
    # X_dev = np.concatenate((STATS_dev, LIWC_dev), axis = 1)

    # X_train = np.concatenate((SBERT_train, STATS_train, LIWC_train), axis = 1)
    # X_dev = np.concatenate((SBERT_dev, STATS_dev, LIWC_dev), axis = 1)

    scaler = StandardScaler().fit(X_train)

    X_train_scale = scaler.transform(X_train)
    X_dev_scale = scaler.transform(X_dev)

    # X_train_scale = np.multiply(np.sign(X_train_scale), np.power(np.abs(X_train_scale), 0.5)) # FOR ELM
    # X_dev_scale = np.multiply(np.sign(X_dev_scale), np.power(np.abs(X_dev_scale), 0.5)) # FOR ELM

    X_train_scale = normalize(X_train_scale)
    X_dev_scale = normalize(X_dev_scale)

    X_train_scale = np.insert(X_train_scale, 0, 1, axis=1)
    X_dev_scale = np.insert(X_dev_scale, 0, 1, axis=1)


    symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep = ';')
    PHQ_sev_train = symptoms_data.loc[56:, 'PHQ_8Total']
    PHQ_sev_dev = symptoms_data.loc[:55, 'PHQ_8Total']

    PHQ_symptoms = symptoms_data.drop(columns = ['Participant_ID', 'AVECParticipant_ID', 'PHQ_8Total']).to_numpy()

    PHQ_symptoms_train = PHQ_symptoms[56:]
    PHQ_symptoms_dev = PHQ_symptoms[:56]

    y_train_symp_unscaled = PHQ_symptoms_train
    y_dev_symp = PHQ_symptoms_dev

    y_train_sev = np.array(PHQ_sev_train)
    y_dev_sev = np.array(PHQ_sev_dev)

    # overall_mean = np.mean(y_train_symp_unscaled, axis = None)
    # overall_sd = np.std(y_train_symp_unscaled, axis = None)

    symptom_means_list = []
    y_train_symp = np.zeros((len(y_train_symp_unscaled), 8))
    for i in range(8):
        mean_i = np.mean(y_train_symp_unscaled[:,i])
        sd_i = np.std(y_train_symp_unscaled[:,i])
        # mean_i = 0
        # sd_i = 1
        # mean_i = overall_mean
        # sd_i = overall_sd
        symptom_means_list.append([mean_i, sd_i])
        y_train_symp[:,i] = (y_train_symp_unscaled[:,i] - mean_i) / sd_i

    symptom_means_list = np.array(symptom_means_list)

    def best_params_elm(C_list, Kernel_list, folds):
        
        val_pred_C_values = []
        for C in C_list:
            
            val_pred_kernel_values = []
            for Kernel in Kernel_list:
                    
                elm = ELM(c = C, kernel = Kernel, is_classification = False, weighted = True)
                # why don't use scaled training data?  
                          
                elm.fit(X_train_scale, y_train_symp)
                symps_pred_dev = elm.predict(X_dev_scale)
                
                for symp in range(8):
                    symps_pred_dev[:,symp] = (symps_pred_dev[:,symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp,0]
                
                # Sanitizing symptoms:
                symps_pred_dev[symps_pred_dev < 0] = 0
                symps_pred_dev[symps_pred_dev > 3] = 3
                
                # Rounding symptoms:
                symps_pred_dev = np.rint(symps_pred_dev)
                
                score_list = []
                score_list_RMSE = []
                score_list_UAR = []
                for i in range(8):
                    # UAR:
                    try:                                      
                        score_UAR_i = recall_score(y_dev_symp[:,i], symps_pred_dev[:,i], average = 'macro')
                    except ValueError:
                        score_UAR_i = 0
                    score_list_UAR.append(score_UAR_i)

                    # RMSE:
                    score_RMSE_i = mean_squared_error(y_dev_symp[:,i], symps_pred_dev[:,i], squared = False)
                    score_list_RMSE.append(score_RMSE_i)
                    
                    # CCC:
                    score_i = concordance_correlation_coefficient(y_dev_symp[:,i], symps_pred_dev[:,i])
                    score_list.append(score_i)
                
                # print(symps_pred_dev)
                
                pred_dev_summation = np.sum(symps_pred_dev, axis = 1)
                
                # Sanitizing final pred:
                pred_dev_summation[pred_dev_summation < 0] = 0
                pred_dev_summation[pred_dev_summation > 24] = 24
                
                summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, pred_dev_summation)
                summation_pred_RMSE = mean_squared_error(y_dev_sev, pred_dev_summation, squared = False)
                
                
                score_average = np.mean(score_list)
                RMSE_average = np.mean(score_list_RMSE)
                UAR_average = np.mean(score_list_UAR)
                
                val_pred_kernel_values.append([score_average, score_list, 
                                            RMSE_average, score_list_RMSE,
                                            summation_pred_CCC, summation_pred_RMSE,
                                            UAR_average, score_list_UAR])
                
            val_pred_C_values.append(val_pred_kernel_values)
        # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. 
        # If you meant to do this, you must specify 'dtype=object' when creating the ndarray. 
        dev_score_table = np.array(val_pred_C_values, dtype=object)
        
        dev_mean_score_table = dev_score_table[:,:,0]
        dev_per_symptoms_score_table = dev_score_table[:,:,1]
        
        dev_mean_RMSE_score_table = dev_score_table[:,:,2]
        dev_per_symptoms_RMSE_score_table = dev_score_table[:,:,3]
            
        dev_summation_CCC_table = dev_score_table[:,:,4]
        dev_summation_RMSE_table = dev_score_table[:,:,5]
        
        dev_mean_UAR_score_table = dev_score_table[:,:,6]
        dev_per_symptoms_UAR_score_table = dev_score_table[:,:,7]
        
        
        ### Optimization Metric ###
        
        # ## For mean Symptom RMSE:
        # best_C_idx = np.where(dev_mean_RMSE_score_table == np.nanmin(dev_mean_RMSE_score_table))[0][0]
        # best_kernel_idx = np.where(dev_mean_RMSE_score_table == np.nanmin(dev_mean_RMSE_score_table))[1][0] 
        
        # ## For mean Symptom UAR:
        # best_C_idx = np.where(dev_mean_UAR_score_table == np.nanmax(dev_mean_UAR_score_table))[0][0]
        # best_kernel_idx = np.where(dev_mean_UAR_score_table == np.nanmax(dev_mean_UAR_score_table))[1][0] 
            
        # For mean Symptom CCC:
        best_C_idx = np.where(dev_mean_score_table == np.nanmax(dev_mean_score_table))[0][0]
        best_kernel_idx = np.where(dev_mean_score_table == np.nanmax(dev_mean_score_table))[1][0]
                   
        # # For Summation RMSE:
        # best_C_idx = np.where(dev_summation_RMSE_table == np.nanmin(dev_summation_RMSE_table))[0][0]
        # best_kernel_idx = np.where(dev_summation_RMSE_table == np.nanmin(dev_summation_RMSE_table))[1][0]
        
        # # For Summation CCC:
        # best_C_idx = np.where(dev_summation_CCC_table == np.nanmax(dev_summation_CCC_table))[0][0]
        # best_kernel_idx = np.where(dev_summation_CCC_table == np.nanmax(dev_summation_CCC_table))[1][0]
        
        dev_best_score = np.round_(dev_mean_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_per_symps = np.round_(dev_per_symptoms_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        
        dev_best_score_RMSE = np.round_(dev_mean_RMSE_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_RMSE_per_symps = np.round_(dev_per_symptoms_RMSE_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        
        dev_best_summation_CCC = np.round_(dev_summation_CCC_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_summation_RMSE = np.round_(dev_summation_RMSE_table[best_C_idx, best_kernel_idx], decimals = 4)
        
        dev_best_score_UAR = np.round_(dev_mean_UAR_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        dev_best_UAR_per_symps = np.round_(dev_per_symptoms_UAR_score_table[best_C_idx, best_kernel_idx], decimals = 4)
        
        
        return C_list[best_C_idx], Kernel_list[best_kernel_idx], dev_best_score, dev_mean_score_table, dev_best_per_symps.reshape(1,-1), dev_best_score_RMSE, dev_best_RMSE_per_symps, dev_best_summation_CCC, dev_best_summation_RMSE, dev_best_score_UAR, dev_best_UAR_per_symps
    # C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 0.0003, 0.003, 0.03, 0.3, 3, 30, 300]
    # C_range = np.arange(1,10,1)

    # C_range = [0.01, 0.1, 0.5, 1,2,3,4,5,6,7,8,9,10,20,40,80,100,200,300,600,1024,2048]

    best_C, best_kernel, dev_best_score, dev_scores, dev_best_per_symps, dev_best_score_RMSE, dev_best_RMSE_per_symps, dev_best_summation_CCC, dev_best_summation_RMSE, dev_best_score_UAR, dev_best_UAR_per_symps = best_params_elm(C_list = C_range, 
                                                                            Kernel_list = kernels, 
                                                                            folds = 5)

    def get_preds(C, Kernel):
        elm = ELM(c = C, kernel = Kernel, is_classification = False, weighted = True)
                                    
        elm.fit(X_train_scale, y_train_symp)
        symps_pred_dev = elm.predict(X_dev_scale)
        
        for symp in range(8):
            symps_pred_dev[:,symp] = (symps_pred_dev[:,symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp,0]
        
        # Sanitizing symptoms:
        symps_pred_dev[symps_pred_dev < 0] = 0
        symps_pred_dev[symps_pred_dev > 3] = 3
        
        # Rounding symptoms:
        symps_pred_dev = np.rint(symps_pred_dev)
        
        score_list = []
        score_list_RMSE = []
        for i in range(8):
            # # UAR:
            # score_i = recall_score(y_dev_symp[:,i], symps_pred_dev[:,i], average = 'macro')
            # score_list.append(score_i)
            # RMSE:
            score_RMSE_i = mean_squared_error(y_dev_symp[:,i], symps_pred_dev[:,i], squared = False)
            score_list_RMSE.append(score_RMSE_i)
            # CCC:
            score_i = concordance_correlation_coefficient(y_dev_symp[:,i], symps_pred_dev[:,i])
            score_list.append(score_i)
        
        print(np.mean(score_list))
        print(np.mean(score_list_RMSE))
        
        pred_dev_summation = np.sum(symps_pred_dev, axis = 1)
        
        # Sanitizing final pred:
        pred_dev_summation[pred_dev_summation < 0] = 0
        pred_dev_summation[pred_dev_summation > 24] = 24

        return symps_pred_dev, pred_dev_summation

    dev_symp_pred, dev_summation_pred = get_preds(best_C, best_kernel)
    # verify_summation_pred_CCC = concordance_correlation_coefficient(y_dev_sev, dev_summation_pred)
    # verify_summation_pred_RMSE = mean_squared_error(y_dev_sev, dev_summation_pred, squared = False)

    summation_pred_CCC = np.round_(concordance_correlation_coefficient(y_dev_sev, dev_summation_pred), decimals = 4)
    summation_pred_MAE = np.round_(mean_absolute_error(y_dev_sev, dev_summation_pred), decimals = 4)
    summation_pred_RMSE = np.round_(mean_squared_error(y_dev_sev, dev_summation_pred, squared = False), decimals = 4)

    print('Top data use percentage:', Data_use_top_percentage)
    print('Best C is:', best_C)
    print('Is ascending order?', ascending_order)
    print('summation_pred_CCC:', summation_pred_CCC)
    print('summation_pred_RMSE', summation_pred_RMSE)

    result_dict['Data_use_top_percentage'].append(Data_use_top_percentage)
    result_dict['RMSE'].append(summation_pred_RMSE)
    result_dict['CCC'].append(summation_pred_CCC)

result_df = pd.DataFrame(result_dict)
result_df.to_excel('result_scores_w_bias_valence_ascending.xlsx', index=False)
# plt.plot(result_df['Data_use_top_percentage'], result_df['RMSE'], label='RMSE')
plt.plot(result_df['Data_use_top_percentage'], result_df['CCC'], label='CCC')
plt.xlabel('Data_use_top_percentage')
plt.ylabel('Score')
plt.legend()
plt.show()