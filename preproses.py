import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df, is_train=True):
    """Preprocess the data for modeling"""
    
    df = df.copy()

    categorical_features = [
        'HomePlanet', 'CryoSleep', 'Destination',
        'VIP', 'Deck', 'Side', 'Age_group'
    ]

    numerical_features = [
        'Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'Cabin_num', 'Group_size',
        'Solo', 'Family_size', 'TotalSpending',
        'HasSpending', 'NoSpending',
        'Age_missing', 'CryoSleep_missing'
    ] + [col for col in df.columns if '_ratio' in col]

    # Fill missing categorical
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')

    # Fill missing numerical
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Save encoders for Streamlit
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    feature_columns = categorical_features + numerical_features

    X = df[feature_columns]

    if is_train:
        y = df['Transported'].astype(int)
        return X, y, feature_columns
    else:
        return X, feature_columns

