# CodeMixToolkit
Languages: Hindi, English, Malyalam, Gujarati, Tamil and Telugu

Install Instructions:
1. Download the dist directory
2. Do pip install 'location of dist'
3. You can import codemix in your code now

### Recommended
- Run the toolkit in an Anaconda environment.

## CODE-MIX GENERATOR - MODIFIED DOCKER IMAGE

- This modified docker image contains API calls to utilise the aligner and codemix-generator functionalities in a simple manner.

### Pull docker image

- Link to docker hub: https://hub.docker.com/r/prakod/gcm-codemix-generator
- Alternatively, use the command 
```
docker pull prakod/gcm-codemix-generator
```

### Installation instructions (after pulling docker image)
```
docker run -p 5000:5000 -p 6000:6000 prakod/codemix-gcm-generator (this can alternatively be done using Docker desktop)
```
- This will create a container based on the Docker image. Get the ID of the container (using the Desktop app or `docker ps`)
- Then run:
```
docker exec -it <container_id> bash
```
- This will create a bash terminal for you to perform operations on the container.
```
conda activate gcm-venv
git clone https://github.com/prashantkodali/CodeMixToolkit.git
```

### Running jupyter notebook

```
jupyter notebook --ip 0.0.0.0 --port 5000 --no-browser --allow-root
```

### Instructions to run the flask API: 

- Ensure you are in the "library" folder

- Run these commands:
 ```
 >>> export FLASK_APP=gcmgenerator
 >>> flask run -h 0.0.0.0 -p 6000
 ```
- (change port and host details as required)



# CSNLI API Service

A FastAPI-based service for language identification and text processing, particularly focused on Hinglish (Hindi-English) text processing.

## CSNLI API Setup

### Prerequisites
- Python 3.7+
- Required Python packages:
  - fastapi
  - uvicorn
  - pydantic
  - requests (for testing)

### Installation

1. Install the required packages:
```bash
pip install fastapi uvicorn pydantic requests
```

2. Make sure the model files are in the correct locations:
   - `lid_models/hinglish`
   - `nmt_models/rom2hin.pt`
   - `nmt_models/eng2eng.pt`

## Running the CSNLI Service

### Development Mode
```bash
# Using Python directly
python csnli_api.py

# Or using uvicorn with auto-reload
uvicorn csnli_api:app --host 0.0.0.0 --port 6001 --reload
```

### Production Mode
```bash
uvicorn csnli_api:app --host 0.0.0.0 --port 6001
```

## CSNLI API Documentation

Once the service is running, you can access:
- Interactive API documentation: `http://localhost:6001/docs`
- Alternative documentation: `http://localhost:6001/redoc`

## CSNLI API Endpoint

### POST /csnli-lid
Processes input text for language identification and normalization.

**Request Body:**
```json
{
    "text": "your text here"
}
```

**Response:**
```json
{
    "csnli_op": {
        "og_text": "original text",
        "text": ["processed", "words"],
        "norm_text": ["normalized", "words"],
        "lid": ["language", "tags"]
    }
}
```

## Testing the CSNLI API

1. Make sure the API service is running
2. Run the test script:
```bash
python test_csnli_api.py
```

The test script includes several test cases:
- Hinglish text
- Pure Hindi text
- Pure English text
- Mixed Hindi-English text
- Another Hinglish example

## CSNLI API Example Usage

### Using curl
```bash
curl -X POST "http://localhost:6001/csnli-lid" \
     -H "Content-Type: application/json" \
     -d '{"text": "i thght mosam dfrnt hoga bs fog h"}'
```

### Using Python
```python
import requests

url = "http://localhost:6001/csnli-lid"
headers = {"Content-Type": "application/json"}
data = {"text": "your text here"}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## CSNLI API Error Handling

The API handles various error cases:
- Invalid input format
- Processing errors
- Server errors

All errors are returned with appropriate HTTP status codes and error messages.

## Contributing

Feel free to submit issues and enhancement requests.
