from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from three_step_decoding import ThreeStepDecoding
from lang_tagger import Meta, LID
from functools import lru_cache
import sys
import os
import pickle

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Create FastAPI app
app = FastAPI(
    title="CSNLI API",
    description="Language Identification and Text Processing API",
    version="1.0.0",
)


# Define request model
class TextRequest(BaseModel):
    text: str


# Define response model
class TextResponse(BaseModel):
    csnli_op: Dict[str, Any]


def load_model():
    """
    Load the model with proper pickling support.
    This function ensures the Meta class is available during unpickling.
    """
    # First, ensure Meta class is in the global namespace
    global Meta
    if "Meta" not in globals():
        from lang_tagger import Meta

    # Now load the model
    return ThreeStepDecoding(
        "lid_models/hinglish",
        htrans="nmt_models/rom2hin.pt",
        etrans="nmt_models/eng2eng.pt",
    )


@lru_cache()
def get_tsd():
    """
    Creates a singleton instance of ThreeStepDecoding.
    The @lru_cache decorator ensures only one instance is created and reused.
    """
    return load_model()


@app.post("/csnli-lid", response_model=TextResponse)
async def process_text(
    request: TextRequest, tsd: ThreeStepDecoding = Depends(get_tsd)
) -> TextResponse:
    """
    Process input text for language identification and normalization.

    Args:
        request: TextRequest containing the input text
        tsd: ThreeStepDecoding instance (injected by FastAPI)

    Returns:
        TextResponse containing processed text with language tags
    """
    try:
        op = tsd.tag_sent(request.text)
    except ValueError:
        op = None
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        op = None

    if op:
        t, nt, l = [], [], []
        for el in op:
            t.append(el[0])
            nt.append(el[1])
            l.append(el[2])
    else:
        t = request.text.split()
        nt = None
        l = None

    response_data = {
        "text_str": request.text,
        "text_tokenized": t,
        "norm_text": nt,
        "lid": l,
    }

    return TextResponse(csnli_op=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)
