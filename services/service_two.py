import logging
from fastapi import HTTPException
from transformers import pipeline
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Create a custom logger
logger = logging.getLogger(__name__)
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')


class ClassificationInput(BaseModel):
    sequence_to_classify: str
    candidate_labels: List[str]

@router.post("/classify-sequence", summary="Classify the given sequence of text")
async def classify_sequence(input: ClassificationInput):
    sequence_to_classify = input.sequence_to_classify
    candidate_labels = input.candidate_labels
    # rest of your code
    """
    Classify the given sequence of text into one of the predefined categories.

    This endpoint accepts a string of text and a list of candidate labels,
    and uses a zero-shot classification model to classify the sequence into one of the candidate categories.

    Parameters:
    - sequence_to_classify: A string of text to be classified.
    - candidate_labels: A list of strings representing the candidate categories.

    Returns:
    - A dictionary with keys `labels` and `scores` indicating the predicted categories and their respective scores.

    Raises:
    - HTTPException: If an error occurs during the classification process, it raises an HTTPException with a status code of 500 and the error message.
    """
    try:
        result = classifier(sequence_to_classify, candidate_labels)
        logger.info('Text classified successfully')
        return result
    except Exception as e:
        logger.error('Error occurred while classifying text: %s', e)
        raise HTTPException(status_code=500, detail=str(e))