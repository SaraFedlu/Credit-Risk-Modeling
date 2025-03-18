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
    try:
        print("Step 1: Initial Data Types\n", df.dtypes)
        df = df.drop(columns=['CurrencyCode', 'CountryCode'])
        print("Step 2: After Dropping Columns\n", df.dtypes)
    except Exception as e:
        print(f"Error dropping columns: {e}")
        return None
    
    try:
        numerical_features = ['Amount', 'Value']
        df = handle_outliers_log_transform(df, numerical_features)
        print("Step 3: After Log Transformation\n", df.dtypes)
    except Exception as e:
        print(f"Error handling outliers and log transformation: {e}")
        return None
    
    try:
        provider_threshold = 500
        df['ProviderId'] = df['ProviderId'].apply(
            lambda x: x if df['ProviderId'].value_counts().get(x, 0) >= provider_threshold else 'Other'
        )
        print("Step 4: After Grouping ProviderId\n", df.dtypes)
    except Exception as e:
        print(f"Error processing ProviderId categories: {e}")
        return None
    
    try:
        category_threshold = 500
        df['ProductCategory'] = df['ProductCategory'].apply(
            lambda x: x if df['ProductCategory'].value_counts().get(x, 0) >= category_threshold else 'Other'
        )
        print("Step 5: After Grouping ProductCategory\n", df.dtypes)
    except Exception as e:
        print(f"Error processing ProductCategory categories: {e}")
        return None
    
    try:
        df['Is_Positive_Amount'] = df['Amount'] > 0
        df['Amount_log'] = np.where(df['Amount'] > 0, np.log1p(df['Amount']), 0)
        print("Step 6: After Feature Engineering\n", df.dtypes)
    except Exception as e:
        print(f"Error creating log-transformed features: {e}")
        return None
    
    try:
        agg_features = df.groupby('CustomerId')['Amount'].agg([
            ('Total_Transaction_Amount', 'sum'),
            ('Avg_Transaction_Amount', 'mean'),
            ('Transaction_Count', 'count'),
            ('Std_Transaction_Amount', 'std')
        ]).reset_index()
        df = df.merge(agg_features, on='CustomerId', how='left')
        print("Step 7: After Aggregation\n", df.dtypes)
    except Exception as e:
        print(f"Error aggregating features: {e}")
        return None
    
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        print("Step 8: After Converting to DateTime\n", df.dtypes)
    except Exception as e:
        print(f"Error converting TransactionStartTime: {e}")
        return None
    
    try:
        categorical_cols = ['ProviderId', 'ProductCategory', 'ChannelId']
        ohe = load('data/ohe_encoder.pkl')
        encoded_feature_names = pd.read_csv('data/encoded_features.csv').squeeze()
        new_encoded = ohe.transform(df[categorical_cols])
        new_encoded_df = pd.DataFrame(new_encoded, columns=ohe.get_feature_names_out())
        new_encoded_df = new_encoded_df.reindex(columns=encoded_feature_names, fill_value=0)
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        df = pd.concat([df[numerical_cols], new_encoded_df], axis=1)
        print("Step 9: After One-Hot Encoding\n", df.dtypes)
    except Exception as e:
        print(f"Error handling one-hot encoding: {e}")
        return None
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Step 10: After Imputing Missing Values\n", df.dtypes)
    except Exception as e:
        print(f"Error imputing missing values: {e}")
        return None
    
    try:
        scaler = StandardScaler()
        numerical_cols = ['Total_Transaction_Amount', 'Avg_Transaction_Amount',
                          'Transaction_Count', 'Std_Transaction_Amount', 'Amount_log', 'Value_log']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print("Step 11: After Normalization\n", df.dtypes)
    except Exception as e:
        print(f"Error normalizing features: {e}")
        return None
    
    try:
        exclude_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
        data_for_binning = df.drop(columns=exclude_columns)
        data_binned = sc.woebin_ply(data_for_binning, bins)
        customer_ids = df[['CustomerId']].loc[data_binned.index]
        data_merged = data_binned.copy()
        data_merged['CustomerId'] = customer_ids['CustomerId']
        data_merged = data_merged.select_dtypes(include=[np.number])
        print("Step 12: After Binning\n", df.dtypes)
    except Exception as e:
        print(f"Error in binning process: {e}")
        return None
    
    return data_merged

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

@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Preprocess the data
        df_processed = preprocess_data(df.copy())

        if df_processed is None:
            raise HTTPException(status_code=500, detail="Preprocessing failed. Check logs.")

        # Ensure only numeric data is passed to the model
        print("Processed DataFrame Types:\n", df_processed.dtypes)

        # Check for non-numeric values
        non_numeric_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            raise HTTPException(status_code=400, detail=f"Non-numeric columns found: {non_numeric_cols.tolist()}")

        # Convert DataFrame to numpy array
        features = df_processed.values.astype(float)  # Explicit conversion to catch errors early

        # Make predictions
        predictions = model.predict(features)

        # Attach predictions to the DataFrame and return HTML table
        df['prediction'] = predictions
        return df.to_html(classes="table table-striped", border=0)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"ValueError: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {e}")