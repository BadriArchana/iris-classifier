# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('species', axis=1)
    y = LabelEncoder().fit_transform(df['species'])
    return train_test_split(X, y, test_size=0.2, random_state=42)
