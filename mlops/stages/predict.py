from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import json

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class EmbedTitle(BaseModel):
    features: list[int | float]

app = FastAPI()

@app.post("/predict")
def predict(embed_title: EmbedTitle):
    if len(embed_title.features) != model.n_features_in_+1:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    prediction = int(model.predict([embed_title.features[:-1]])[0])
    return {'prediction':prediction, 'real_label':embed_title.features[-1]}

@app.get("/")
def read_root():
    return {"message": "Title grouping model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)