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
import math
import librosa
from sklearn.manifold import TSNE

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

# Sbert_file = '/home/legalalien/Documents/Jiawei/Filtering_audio_text_project/Features/SBERT_train_dev_test.pkl'
Sbert_file = '/home/legalalien/Documents/Jiawei/Filtering_audio_text_project/Features/SBERT_miniLM_L6_v2_train_dev_test.pkl'
Sentiment_scores_file = '/home/legalalien/Documents/Jiawei/Filtering_audio_text_project/Features/Twitter_sentiment_score_train_dev_test.pkl'

def apply_cca(X, Y, n_components=1, Xdev=None, Ydev=None):
    import numpy as np
    from sklearn.cross_decomposition import CCA


    # Initialize the CCA model
    cca = CCA(n_components=n_components)

    # Fit the model to the data
    cca.fit(X, Y)
    # Transform the data
    X_c, Y_c = cca.transform(X, Y)

    # Calculate the canonical correlations
    canonical_correlations_train = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_components)

    print("Canonical correlations on train:", canonical_correlations_train)
    
    # Transform the data
    Xdev_c, Ydev_c = cca.transform(Xdev, Ydev)

    # Calculate the canonical correlations
    canonical_correlations_dev = np.corrcoef(Xdev_c.T, Ydev_c.T).diagonal(offset=n_components)
    
    print("Canonical correlations on dev:", canonical_correlations_dev)
    return canonical_correlations_train, canonical_correlations_dev
    
def get_embedding_and_labell(VAD, map_label, PDEM_w2v2, feature_index, percentage, ascending_order=False):
        
    ## Convert scores_data to DataFrame and set the ID column as the index
    df_scores = pd.DataFrame(VAD, columns=['arousal', 'valence', 'dominance'])
    df_scores.set_index(map_label['Wav-path'], inplace=True)

    # Merge the two DataFrames
    df_combined = pd.DataFrame(map_label, columns=['Wav-path','AVECParticipant_ID','Duration','PHQ_8Total'])
    df_combined['embedding_feature'] = [np.array(embedding) for embedding in PDEM_w2v2]
    
    # Merge the two DataFrames based on the Wav-path index
    df_combined = df_combined.merge(df_scores, left_on='Wav-path', right_index=True)
    # df_combined['Absolute_btw_arousal_valence'] = abs(df_combined['arousal'] - df_combined['valence'])

    # Removing chunks that duration less than 1 second.
    df_combined_gt_1s = df_combined[df_combined['Duration'] >= 0.5]

    # Calculate the mean of arousal, valence and dominance for each participant
    mean_arousal_valence = df_combined_gt_1s.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].transform('mean')

    # Adding new column 'arousal_diff' and 'valence_diff'
    df_combined_gt_1s['arousal_diff'] = abs(df_combined_gt_1s['arousal'] - mean_arousal_valence['arousal'])
    df_combined_gt_1s['valence_diff'] = abs(df_combined_gt_1s['valence'] - mean_arousal_valence['valence'])
    df_combined_gt_1s.loc[:, 'dominance_diff'] = abs(df_combined_gt_1s['dominance'] - mean_arousal_valence['dominance'])
    
    # Sort the DataFrame by 'AVECParticipant_ID' and 'Absolute_btw_arousal_valence'
    df_combined_sorted = df_combined_gt_1s.sort_values(by=['AVECParticipant_ID', feature_index], ascending=[True, ascending_order])

    # df_combined_sorted = df_combined.sort_values(by=['AVECParticipant_ID'], ascending=[True])
    
    # Delete the data points that arousal or valence score is less than 0
    df_combined_sorted = df_combined_sorted[df_combined_sorted['arousal'] >= 0]
    df_combined_sorted = df_combined_sorted[df_combined_sorted['valence'] >= 0]
    
    # Group by 'AVECParticipant_ID' and get the top percentage of the ranked DataFrame
    if percentage == 100:
        df_top_percentage = df_combined_sorted.groupby('AVECParticipant_ID').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100)))
    else:
        df_top_percentage = df_combined_sorted.groupby('AVECParticipant_ID').apply(lambda x: x.head(math.ceil(x.shape[0] * percentage / 100))).drop('AVECParticipant_ID', axis=1)
    # df_top_percentage = df_top_percentage.drop('AVECParticipant_ID', axis=1, inplace=True)
    
    # Add a new column in df_combined_sorted named 'Selected', with boolean values. Compare the 'Wav-path' in df_combined_sorted and df_top_percentage. If 'Wav-path' in df_top_percentage is in df_combined_sorted, set 'Selected' to True, otherwise set it to False.
    df_combined_sorted['Selected'] = df_combined_sorted['Wav-path'].isin(df_top_percentage['Wav-path'])
    # Check the number of True and False in 'Selected'
    print(df_combined_sorted['Selected'].value_counts())
    
    # Calculate the mean to get session-level embedding by averaging the embeddings
    avg_embedding = df_top_percentage.groupby('AVECParticipant_ID')['embedding_feature'].apply(lambda x: np.mean(x.values, axis=0))
    avg_scores = df_top_percentage.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].mean()
    std_embedding = df_top_percentage.groupby('AVECParticipant_ID')['embedding_feature'].apply(lambda x: np.std(x.values, axis=0, ddof=0))
    std_scores = df_top_percentage.groupby('AVECParticipant_ID')[['arousal', 'valence', 'dominance']].std(ddof=0)
    
    avg_embedding_unfilter = df_combined_sorted.groupby('AVECParticipant_ID')['embedding_feature'].apply(lambda x: np.mean(x.values, axis=0))
    
    X_unfiltered_utterance = np.vstack(df_combined_sorted['embedding_feature'].to_numpy())
    X_filtered_utterance = np.vstack(df_top_percentage['embedding_feature'].to_numpy())
    Y_unfiltered_utterance_label = df_combined_sorted['PHQ_8Total']
    Y_filtered_utterance_label = df_top_percentage['PHQ_8Total']
    
    ave_embedding_np = np.vstack(avg_embedding.to_numpy())  # to two-dimensional matrix
    avg_scores_np = np.vstack(avg_scores.to_numpy())

    std_embedding_np = np.vstack(std_embedding.to_numpy())  # to two-dimensional matrix
    std_scores_np = np.vstack(std_scores.to_numpy())
    
    avg_std_concat = np.concatenate((ave_embedding_np, std_embedding_np), axis=1)
    
    X_unfiltered_session = np.vstack(avg_embedding_unfilter.to_numpy())
    X_filtered_session = np.vstack(avg_std_concat)
    Y_unfiltered_session_label = df_combined_sorted.groupby('AVECParticipant_ID')['PHQ_8Total'].mean()
    Y_filtered_session_label = df_top_percentage.groupby('AVECParticipant_ID')['PHQ_8Total'].mean()
    
    return df_combined_sorted['Selected'], X_unfiltered_utterance, X_filtered_utterance, Y_unfiltered_utterance_label, Y_filtered_utterance_label, \
         X_filtered_session, Y_filtered_session_label
    
def call_cca():
    # Load VAD features
    VAD_train = pickle.load(open(VAD_train_file, 'rb'))
    VAD_dev = pickle.load(open(VAD_dev_file, 'rb'))
    VAD_test = pickle.load(open(VAD_test_file, 'rb'))

    # Load mapped labels
    train_mapped_label = pd.read_csv(train_mapped_label_file)
    dev_mapped_label = pd.read_csv(dev_mapped_label_file)
    test_mapped_label = pd.read_csv(test_mapped_label_file)
    
    # Load PDEM_w2v2 features
    PDEM_w2v2_train = pickle.load(open(PDEM_w2v2_train_file, 'rb')).to_numpy()
    PDEM_w2v2_dev = pickle.load(open(PDEM_w2v2_dev_file, 'rb')).to_numpy()
    PDEM_w2v2_test = pickle.load(open(PDEM_w2v2_test_file, 'rb')).to_numpy()

    # Draw session-level best filtered embeddings and full embeddings for dev
    # Draw t-SNE for SBERT features for dev
    
    # Filtered VA scores
    feature_index = 'dominance_diff'
    ascending_order = False
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  
    correlations_triandev = []
    for percentage in percentages:
    
        dev_x_unfilter, dev_x_filter, dev_y_unfilter, dev_y_filter, dev_x_unfilter_session, dev_x_filter_session, dev_y_unfilter_session, dev_y_filter_session=get_embedding_and_labell(VAD_dev, dev_mapped_label, PDEM_w2v2_dev, feature_index, percentage, ascending_order)
        train_x_unfilter, train_x_filter, train_y_unfilter, train_y_filter, train_x_unfilter_session, train_x_filter_session, train_y_unfilter_session, train_y_filter_session=get_embedding_and_labell(VAD_train, train_mapped_label, PDEM_w2v2_train, feature_index, percentage, ascending_order)
        # test_x_unfilter, test_x_filter, test_y_unfilter, test_y_filter=get_embedding_and_labell(VAD_test, test_mapped_label, PDEM_w2v2_test, feature_index, percentage, ascending_order) 
        # Session level
        print("Filtered session-level results on percentage ", percentage)
        canonical_correlations_train, canonical_correlations_dev = apply_cca(train_x_filter_session, train_y_filter_session, 1, dev_x_filter_session, dev_y_filter_session)
        correlations_triandev.append([canonical_correlations_train, canonical_correlations_dev])
    
    print(correlations_triandev)

call_cca()