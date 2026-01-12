import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

# --- Helper Functions ---

def fill_None(tr):
    columns_with_none = ['email', 'gender', 'email_verified', 'blue_tick', 'platform', 'embedded_content']
    tr[columns_with_none] = tr[columns_with_none].fillna('Unknown')
    return tr

def determine_activity(row):
    max_activity = max(row['num_followers'], row['num_follow'])
    if max_activity == 0:
        return '1'
    elif 0 < max_activity < 10:
        return '2'
    else:
        return '3'

def variable_conversions(tr):
    tr['num_followers'] = tr['date_of_new_follower'].apply(lambda x: len(x))
    tr['num_follow'] = tr['date_of_new_follow'].apply(lambda x: len(x))
    tr['user_activity'] = tr.apply(determine_activity, axis=1)
    return tr

def text_ext(tr):
    """
    Extracts positive/negative word ratios based on training data.
    """
    # Note: Requires 'sentiment' column to be present (only for training set)
    tr['tokenized_text'] = tr['text'].apply(lambda x: word_tokenize(x.lower()))
    
    # Simple logic to find popular words (simplified from original for brevity)
    # In a real pipeline, this should be fitted on train and applied to test.
    most_popular_words = pd.Series(' '.join(tr['text']).lower().split()).value_counts()[:2000].index.tolist()
    
    l = []
    for word in most_popular_words:
        escaped_word = re.escape(word)
        df_word_positive = tr[tr['text'].str.contains(escaped_word) & (tr['sentiment'] == 'positive')]
        df_word_negative = tr[tr['text'].str.contains(escaped_word) & (tr['sentiment'] == 'negative')]
        
        count_pos = len(df_word_positive)
        count_neg = len(df_word_negative)
        
        if count_pos == 0 and count_neg == 0:
            continue
            
        new_row = {'Word': word, 
                   'Positve_Ratio': count_pos/(count_pos+count_neg), 
                   'Negativee_Ratio': count_neg/(count_pos+count_neg)}
        l.append(new_row)
        
    word_counts_df = pd.DataFrame(l)
    Pos_words = word_counts_df[word_counts_df['Positve_Ratio'] >= 0.75]
    Neg_words = word_counts_df[word_counts_df['Negativee_Ratio'] >= 0.75]
    
    return Pos_words, Neg_words

def FeatureExtraction(tr):
    tr['message_date'] = pd.to_datetime(tr['message_date'])
    tr['message_hour'] = tr['message_date'].dt.hour
    
    tr['account_creation_date'] = pd.to_datetime(tr['account_creation_date'])
    tr['user_seniority'] = (tr['message_date'] - tr['account_creation_date']).dt.days / 365
    
    tr['previous_messages_num'] = tr['previous_messages_dates'].apply(lambda x: len(x))
    
    tr['email_org'] = tr['email'].str.split('@').str[1].str.split('.').str[0]
    tr['email_org'] = tr['email_org'].fillna('Unknown')
    
    tr['word_count'] = tr['text'].apply(lambda x: len(x.split()))
    return tr

def adjust_previous_messages_num(value):
    if value >= 10: return 10
    else: return value

def FeatureRepresentation(tr):
    bins = [0, 6, 7, 8, 9, 10, float('inf')]
    labels = ['1', '2', '3', '4', '5', '6']
    tr['user_seniority_range'] = pd.cut(tr['user_seniority'], bins=bins, labels=labels, right=False)
    
    tr['previous_messages_num'] = tr['previous_messages_num'].apply(adjust_previous_messages_num)
    
    gender_mapping = {'M': 1, 'F': 2, 'Unknown': 3}
    tr['gender'] = tr['gender'].replace(gender_mapping)
    
    email_verified_mapping = {False: 1, True: 2, 'Unknown': 3}
    tr['email_verified'] = tr['email_verified'].replace(email_verified_mapping)
    
    blue_tick_mapping = {False: 1, True: 2, 'Unknown': 3}
    tr['blue_tick'] = tr['blue_tick'].replace(blue_tick_mapping)
    
    tr = pd.get_dummies(tr, columns=['embedded_content', 'platform'])
    
    encoder = LabelEncoder()
    # Note: Ideally encoder should be fitted on train and transformed on test
    tr['email_org_encoded'] = encoder.fit_transform(tr['email_org'])
    
    return tr

def FeatureSelection(tr, y_train, k=19):
    fisher_selector = SelectKBest(score_func=f_classif, k=k)
    fisher_selector.fit(tr, y_train)
    
    cols = fisher_selector.get_support(indices=True)
    X_top_features = tr.iloc[:, cols]
    feature_names = tr.columns[cols].tolist() # Save names for validation set
    
    return X_top_features, feature_names

def filterColumns(tr):
    cols_to_drop = ['text', 'textID', 'message_date', 'account_creation_date', 
                    'previous_messages_dates', 'date_of_new_follower', 
                    'date_of_new_follow', 'email', 'user_seniority', 'email_org']
    # Only drop columns that exist
    existing_cols = [c for c in cols_to_drop if c in tr.columns]
    tr.drop(columns=existing_cols, axis=1, inplace=True)
    return tr

# --- Main Pipeline Functions ---

def load_and_split_data(path):
    tr = pd.read_pickle(path)
    X = tr.drop(columns=['sentiment'])
    y_train = pd.Series(tr['sentiment'])
    sentiment_mapping = {'positive': 1, 'negative': 0}
    y = y_train.apply(lambda x: sentiment_mapping.get(x))
    
    # 80-20 split
    return train_test_split(X, y, test_size=0.2, random_state=123)

def apply_preprocessing_train(df, y_train):
    """Pipeline for Training Data"""
    df = fill_None(df)
    df = variable_conversions(df)
    df = FeatureExtraction(df)
    df = FeatureRepresentation(df)
    df = filterColumns(df)
    
    # Handle NaNs created by operations if any (Simple fill for robustness)
    df = df.fillna(0) 
    
    df_selected, feature_names = FeatureSelection(df, y_train)
    return df_selected, feature_names

def apply_preprocessing_test(df, feature_names):
    """Pipeline for Test/Validation Data (uses features selected in train)"""
    df = fill_None(df)
    df = variable_conversions(df)
    df = FeatureExtraction(df)
    df = FeatureRepresentation(df)
    df = filterColumns(df)
    
    df = df.fillna(0)
    
    # Keep only the features selected during training
    # Ensure all columns exist (add missing as 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_names]
