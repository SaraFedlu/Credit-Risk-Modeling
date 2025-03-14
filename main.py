from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from xverse.transformer import WOE
import scorecardpy as sc
import io
import pickle
from joblib import load
from pydantic import BaseModel

app = FastAPI()

# Load the trained model
try:
    model = load("models/random_forest_model.pkl")
    with open('data/woe_bins.pkl', 'rb') as f:
        bins = pickle.load(f)

except Exception as e:
    raise RuntimeError("Model file not found or could not be loaded") from e

def handle_outliers_log_transform(df, columns):
    for col in columns:
        # Add 1 to avoid log(0)
        df[col + '_log'] = np.log1p(df[col])
    return df

# Preprocessing function that mimics steps from the notebooks
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['CurrencyCode', 'CountryCode'])
    # Apply log transformation to Amount and Value
    numerical_features = ['Amount', 'Value']
    df = handle_outliers_log_transform(df, numerical_features)

    # Group infrequent ProviderId categories
    provider_threshold = 500
    df['ProviderId'] = df['ProviderId'].apply(
        lambda x: x if df['ProviderId'].value_counts()[x] >= provider_threshold else 'Other'
    )

    # Group infrequent ProductCategory categories
    category_threshold = 500
    df['ProductCategory'] = df['ProductCategory'].apply(
        lambda x: x if df['ProductCategory'].value_counts()[x] >= category_threshold else 'Other'
    )

    df['Is_Positive_Amount'] = df['Amount'] > 0
    df['Amount_log'] = np.where(df['Amount'] > 0, np.log1p(df['Amount']), 0)
    
    agg_features = df.groupby('CustomerId')['Amount'].agg([
        ('Total_Transaction_Amount', 'sum'),
        ('Avg_Transaction_Amount', 'mean'),
        ('Transaction_Count', 'count'),
        ('Std_Transaction_Amount', 'std')
    ]).reset_index()

    df = df.merge(agg_features, on='CustomerId', how='left')

    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Extract features
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    categorical_cols = ['ProviderId', 'ProductCategory', 'ChannelId']
    
    # Load saved artifacts
    ohe = joblib.load('data/ohe_encoder.pkl')
    encoded_feature_names = pd.read_csv('data/encoded_features.csv').squeeze()

    # Transform new data
    new_encoded = ohe.transform(df[categorical_cols])
    new_encoded_df = pd.DataFrame(new_encoded, columns=ohe.get_feature_names_out())

    # Ensure column consistency
    new_encoded_df = new_encoded_df.reindex(columns=encoded_feature_names, fill_value=0)

    # Combine with non-categorical features
    numerical_cols = [col for col in df.columns if col not in categorical_cols]
    df = pd.concat([df[numerical_cols], new_encoded_df], axis=1)

    # Impute missing numeric values using the median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Normalize numeric features
    scaler = StandardScaler()
    numerical_cols = ['Total_Transaction_Amount', 'Avg_Transaction_Amount',
                  'Transaction_Count', 'Std_Transaction_Amount', 'Amount_log', 'Value_log']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    exclude_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
    data_for_binning = df.drop(columns=exclude_columns)

    data_binned = sc.woebin_ply(data_for_binning, bins)
    low_iv_features_woe = ['ProviderId_ProviderId_3', 'ProductCategory_Other', 'ProductCategory_tv', 'ChannelId_ChannelId_5', 'ProviderId_Other', 'ProductCategory_data_bundles', 'ProductCategory_utility_bill', 'ChannelId_ChannelId_1']
    data_binned.drop(columns=low_iv_features_woe, inplace=True)

    customer_ids = df[['CustomerId']]
    customer_ids = customer_ids.loc[data_binned.index]

    # Merge CustomerId back into the data_binned subset
    data_merged = data_binned.copy()
    data_merged['CustomerId'] = customer_ids['CustomerId']

    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    # Recency: Days since the last transaction
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency['Recency'] = (df['TransactionStartTime'].max() - recency['TransactionStartTime']).dt.days
    recency.drop(columns='TransactionStartTime', inplace=True)

    # Frequency: Total transactions per customer
    frequency = df.groupby('CustomerId')['TransactionId'].count().reset_index()
    frequency.rename(columns={'TransactionId': 'Frequency'}, inplace=True)

    # Monetary: Total and average transaction amounts
    monetary = df.groupby('CustomerId')['Amount'].agg(['sum', 'mean']).reset_index()
    monetary.rename(columns={'sum': 'Monetary_Total', 'mean': 'Monetary_Avg'}, inplace=True)

    # Combine RFMS metrics
    rfms = recency.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')

    # Merge RFMS metrics into data_merged
    data_merged = data_merged.merge(rfms, on='CustomerId', how='left')
    return data_binned

# Simple UI for file upload
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>Credit Risk Model Prediction</title>
        </head>
        <body>
            <h2>Upload CSV File for Prediction</h2>
            <form action="/uploadfile/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    return content

# Endpoint to process the uploaded file, preprocess it, and return predictions
@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Preprocess the data
        df_processed = preprocess_data(df.copy())
        
        # Make predictions
        features = df_processed.values
        predictions = model.predict(features)
        
        # Attach predictions to the DataFrame and convert to HTML table
        df['prediction'] = predictions
        return df.to_html(classes="table table-striped", border=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
