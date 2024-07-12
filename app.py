import io
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

from run import Forecasting

# Load the trained model
F = Forecasting(
    model_name='ConvLSTM',
    model_id='ConvLSTM_carbon',
    data_path="dk_dk2_clean.csv",
    n_past=168,
    n_future=24,
    n_feature=50,
    batch_size=64,
    epochs=50,
    learning_rate=0.0001,
    n_predict=1
)
model = F.load_model()  ##Load the pre-trained model
F.ts_cv(4)              ## setup the scaler

app = FastAPI()

class PredictRequest(BaseModel):
    data: str

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(io.StringIO(file.file.read().decode('utf-8')))
    
    # Ensure df matches the expected input format
    if df.shape[1] != F.n_feature:
        return {"error": "Invalid input dimensions"}
    
    # Preprocess the data
    scaler = F.scaler
    x = scaler.transform(df.values)
    x=x.reshape(1,x.shape[0],x.shape[1])   ## reshape x
    x = torch.tensor(x).float().to(F.device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        outputs = outputs.detach().cpu().numpy()
    
    if F.scale:
        outputs = F.inverse_transform(np.tile((outputs.squeeze(0)), (1, F.n_feature)))
        outputs = outputs[:, -F.n_predict]
    
    return {"predictions": outputs.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
