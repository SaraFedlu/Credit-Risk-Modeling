from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from joblib import load
import pickle
import io

app = FastAPI()

# Load the trained model (ensure the model file exists at the given path)
try:
    model = load("models/random_forest_model.pkl")
except Exception as e:
    raise RuntimeError("Model file not found or could not be loaded") from e

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

# Endpoint to process the uploaded file and return predictions
@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Read uploaded file content
        contents = await file.read()
        # Create a file-like object from the bytes
        df = pd.read_csv(io.BytesIO(contents))
        
        # each row in the CSV corresponds to one instance with features matching the model's expectations
        features = df.values
        
        # Get predictions from the model
        predictions = model.predict(features)
        df["prediction"] = predictions
        
        # Convert the dataframe with predictions to HTML table for display
        return df.to_html(classes="table table-striped", border=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))