from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from three_step_decoding import ThreeStepDecoding
from lang_tagger import Meta

# Initialize the ThreeStepDecoding model
tsd = ThreeStepDecoding(
    'lid_models/hinglish',
    htrans='nmt_models/rom2hin.pt',
    etrans='nmt_models/eng2eng.pt'
)


# Create FastAPI app
app = FastAPI(
    title="CSNLI API",
    description="Language Identification and Text Processing API",
    version="1.0.0"
)

# Define request model
class TextRequest(BaseModel):
    text: str

# Define response model
class TextResponse(BaseModel):
    csnli_op: Dict[str, Any]

@app.post("/csnli-lid", response_model=TextResponse)
async def process_text(request: TextRequest) -> TextResponse:
    """
    Process input text for language identification and normalization.
    
    Args:
        request: TextRequest containing the input text
        
    Returns:
        TextResponse containing processed text with language tags
    """
    try:
        op = tsd.tag_sent(request.text)
    except ValueError:
        op = None
    except Exception:
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
        "lid": l
    }
    
    return TextResponse(csnli_op=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000) 