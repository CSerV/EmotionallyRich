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
import warnings
import seaborn as sns

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
test_mapped_label_file = '/media/legalalien/Data2/AVEC2019_DAIC_WOZ/split_audio_labels/detailed_test.csv'

# Load features files
PDEM_w2v2_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_train/w2v2_train.pkl'
PDEM_w2v2_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_dev/w2v2_dev.pkl'
PDEM_w2v2_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_test/w2v2_test.pkl'
VAD_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_train/vad_train.pkl'
VAD_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_dev/vad_dev.pkl'
VAD_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_test/vad_test.pkl'

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
    # adding new column 'arousal_diff' and 'valence_diff'
    df_combined_gt_1s['arousal_diff'] = abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s['valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence'])
    df_combined_gt_1s.loc[:, 'dominance_diff'] = abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])

    # adding new column 'arousal_valence_diff'
    df_combined_gt_1s['arousal_valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s.loc[:, 'arousal_dominance_diff'] = abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s.loc[:, 'valence_dominance_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])
    df_combined_gt_1s.loc[:, 'arousal_valence_dominance_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence']) + abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal']) + abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])

    # Sort the DataFrame by 'AVECParticipant_ID' and 'Absolute_btw_arousal_valence'
    df_combined_sorted = df_combined_gt_1s.sort_values(by=['AVECParticipant_ID', feature_index], ascending=[True, ascending_order])

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
    # For Pandas pkg, ddof default value of std() is 0
    std_scores = df_top_percentage.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].std(ddof=0)
    # Convert to numpy array
    ave_embedding_np = np.vstack(avg_embedding.to_numpy())  # to two-dimensional matrix
    avg_scores_np = np.vstack(avg_scores.to_numpy())
    std_embedding_np = np.vstack(std_embedding.to_numpy())  # to two-dimensional matrix
    std_scores_np = np.vstack(std_scores.to_numpy())

    print("Average Wav2Vec Embedding for Top", percentage, "% absolute_btw_arousal_valence:")
    print("\nAverage Scores for Top", percentage, "% absolute_btw_arousal_valence:")
    print(avg_scores)
    return ave_embedding_np, avg_scores_np, std_embedding_np, std_scores_np



Data_use_top_percentage = 100
# C is chosen by the dev set
C = 10
ascending_order = False
feature_index = 'dominance_diff'

dev_avg_embedding, dev_avg_scores, dev_std_embedding, dev_std_scores = get_chosen_features(dev_mapped_label_file, PDEM_w2v2_dev_file, VAD_dev_file, Data_use_top_percentage, feature_index, ascending_order)
train_avg_embedding, train_avg_scores, train_std_embedding, train_std_scores = get_chosen_features(train_mapped_label_file, PDEM_w2v2_train_file, VAD_train_file, Data_use_top_percentage, feature_index, ascending_order)
test_avg_embedding, test_avg_scores, test_std_embedding, test_std_scores = get_chosen_features(test_mapped_label_file, PDEM_w2v2_test_file, VAD_test_file, Data_use_top_percentage, feature_index, ascending_order)

X_train = np.concatenate((train_avg_embedding, train_std_embedding), axis=1)
X_dev = np.concatenate((dev_avg_embedding, dev_std_embedding), axis=1)
X_test = np.concatenate((test_avg_embedding, test_std_embedding), axis=1)

X_train = np.concatenate((X_dev, X_train), axis = 0)

scaler = StandardScaler().fit(X_train)

X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# X_train_scale = np.multiply(np.sign(X_train_scale), np.power(np.abs(X_train_scale), 0.5)) # FOR ELM
# X_test_scale = np.multiply(np.sign(X_test_scale), np.power(np.abs(X_test_scale), 0.5)) # FOR ELM

X_train_scale = normalize(X_train_scale)
X_test_scale = normalize(X_test_scale)

X_train_scale = np.insert(X_train_scale, 0, 1, axis=1)
X_test_scale = np.insert(X_test_scale, 0, 1, axis=1)

symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep = ';')
PHQ_sev_train = symptoms_data.loc[:,'PHQ_8Total']

PHQ_symptoms_train = symptoms_data.drop(columns = ['Participant_ID', 'AVECParticipant_ID', 'PHQ_8Total']).to_numpy()

test_data = pd.read_csv('test_split.csv', sep = ',')

y_train_symp_unscaled = PHQ_symptoms_train

y_train_sev = np.array(PHQ_sev_train)
y_test_sev = test_data.loc[:, 'PHQ_Score'].to_numpy()

symptom_means_list = []
y_train_symp = np.zeros((len(y_train_symp_unscaled), 8))
for i in range(8):
    mean_i = np.mean(y_train_symp_unscaled[:,i])
    sd_i = np.std(y_train_symp_unscaled[:,i])
    # mean_i = 0
    # sd_i = 1
    symptom_means_list.append([mean_i, sd_i])
    y_train_symp[:,i] = (y_train_symp_unscaled[:,i] - mean_i) / sd_i

symptom_means_list = np.array(symptom_means_list)


elm = ELM(c = C, kernel = 'linear', is_classification = False, weighted = True)

elm.fit(X_train_scale, y_train_symp)
symps_pred_test = elm.predict(X_test_scale)
# np.savez('symps_pred_test_SBERT.npz',symps_pred_test_SBERT = symps_pred_test)

for symp in range(8):
    symps_pred_test[:,symp] = (symps_pred_test[:,symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp,0]

# # Sanitizing symptoms:
# symps_pred_test[symps_pred_test < 0] = 0
# symps_pred_test[symps_pred_test > 3] = 3

# Computing sum of symptoms
pred_test_summation = np.sum(symps_pred_test, axis = 1)

# Sanitizing symptoms:
pred_test_summation[pred_test_summation < 0] = 0
pred_test_summation[pred_test_summation > 24] = 24

# Computing scores, with NaN instances included
# summation_pred_CCC = np.round_(concordance_correlation_coefficient(y_test_sev, pred_test_summation), decimals = 4)
# summation_pred_MAE = np.round_(mean_absolute_error(y_test_sev, pred_test_summation), decimals = 4)
# summation_pred_RMSE = np.round_(mean_squared_error(y_test_sev, pred_test_summation, squared = False), decimals = 4)

nan_instances = [36,40]
# Excluding NaNs (the two test instances which the AVEC'19 competetion has left out), 
# then compute scores:
y_test_sev = np.delete(y_test_sev, nan_instances, axis = 0)
pred_test_summation = np.delete(pred_test_summation, nan_instances, axis = 0)

# Computing scores, with NaN instances included
summation_pred_CCC = np.round_(concordance_correlation_coefficient(y_test_sev, pred_test_summation), decimals = 4)
summation_pred_MAE = np.round_(mean_absolute_error(y_test_sev, pred_test_summation), decimals = 4)
summation_pred_RMSE = np.round_(mean_squared_error(y_test_sev, pred_test_summation, squared = False), decimals = 4)

print('Top data use percentage:', Data_use_top_percentage)
print('summation_pred_CCC:', summation_pred_CCC)
print('summation_pred_RMSE', summation_pred_RMSE)

df_pred = pd.DataFrame({'Actual PHQ8 Score': y_test_sev, 'Predicted Score': pred_test_summation})
sns.set(font_scale = 1.2)
g = sns.relplot(data = df_pred, x = 'Actual PHQ8 Score',  y = 'Predicted Score', color = 'b')
g.ax.axline(xy1=(0, 0), slope=1, dashes=(3, 2))
g.ax.set(ylim=(-1, 25), xlim = (-1,25))

plt.scatter(y_test_sev, pred_test_summation)
plt.show()

# df_dev_pred_symps_regression = pd.DataFrame(symps_pred_test)
# df_dev_pred_symps_regression.to_csv('Multi-task regression symptom predictions TEST.csv')