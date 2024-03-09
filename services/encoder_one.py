from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from transformers import LayoutLMv3Model, LayoutLMv3Tokenizer
import torch
import logging

router = APIRouter()


class ClassifyTextRequest(BaseModel):
    text: str


# Initialize a LayoutLMv3 model and tokenizer
model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/classify-text", summary="Classify the given text")
async def classify_text(request: ClassifyTextRequest):
    text = request.text
    logger.info(f"Received text for classification: {text}")
    """
    Classify the given text into a category.

    This endpoint accepts a string of text, tokenizes it, and uses a pre-trained model
    to classify the text into one of several predefined categories.

    Parameters:
    - text: A string of text to classify.

    Returns:
    - A dictionary with a key `predicted_class` indicating the predicted category of the text.
    """

    # Split the input text into words
    words = text.split()

# Tokenize the input words
    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
    # Tokenize the input text
    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform text classification
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.last_hidden_state.mean(dim=1)).item()

    return {"predicted_class": predicted_class}


@router.get("/heartbeat", summary="Check if the server is running")
async def heartbeat():
    """
    Check if the server is running.

    This endpoint responds to a GET request and returns a simple message indicating that the server is running.

    Returns:
    - A dictionary with a key `status` and a value of "Running".
    """
    return {"status": "Running"}


@router.post("/classify-document", summary="Classify the content of an uploaded document")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify the content of an uploaded document.

    This endpoint accepts a document file, reads its content, tokenizes the text,
    and uses a pre-trained model to classify the document into a predicted category.

    Parameters:
    - file: An uploaded document file.

    Returns:
    - A dictionary with a key `predicted_class` indicating the classified category of the document.
    """
    logger.info("Received a document for classification")

    # Read the uploaded document
    text = await file.read()
    text = text.decode("utf-8")
    words = text.split()

    # Tokenize the document text
    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)

    # Perform document classification
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits.mean(dim=1)).item()

    logger.info(f"Document classified into class {predicted_class}")

    return {"predicted_class": predicted_class}
