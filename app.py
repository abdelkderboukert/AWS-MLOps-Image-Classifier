from fastapi import FastAPI, File, UploadFile
from src.preprocessing import preprocess_image
from src.utils import get_predicted_keywords, generate_caption
import uvicorn
import joblib

app = FastAPI()

# Load your pre-trained model and binarizer
model = joblib.load("models/best_model.pkl")
mlb = joblib.load("models/mlb.pkl")
extractor = joblib.load("models/extractor.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image bytes
    contents = await file.read()
    
    # 2. Preprocess using your existing logic
    img = preprocess_image(contents)
    
    # 3. Extract features and Predict
    features = extractor.transform([img])
    keywords = get_predicted_keywords(model, features, mlb)
    caption = generate_caption(keywords)
    
    return {"caption": caption, "tags": keywords}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)