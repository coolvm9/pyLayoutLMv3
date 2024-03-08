from fastapi import FastAPI, UploadFile, File
from transformers import LayoutLMv3Model, LayoutLMv3Tokenizer
import torch

app = FastAPI()

# Initialize a LayoutLMv3 model and tokenizer
model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base-uncased")
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base-uncased")

@app.post("/classify-text")
async def classify_text(text: str):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform text classification
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.last_hidden_state.mean(dim=1)).item()

    return {"predicted_class": predicted_class}

@app.post("/classify-document")
async def classify_document(file: UploadFile = File(...)):
    # Read the uploaded document
    text = file.file.read().decode("utf-8")

    # Tokenize the document text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform document classification
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.last_hidden_state.mean(dim=1)).item()

    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
