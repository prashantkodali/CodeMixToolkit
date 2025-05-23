# CSNLI API Docker Setup

This repository contains a Docker setup for the CSNLI (Code-Switched Natural Language Identification) API.

## Quick Start

### Pull the Docker Image
```bash
docker pull prakod/csnli-api
```

### Run in Background (Recommended)
```bash
docker run -d -p 6000:6000 prakod/csnli-api
```

### Run in Foreground (for debugging)
```bash
docker run -p 6000:6000 prakod/csnli-api
```

## Managing the Container

### View Running Containers
```bash
docker ps
```

### View Container Logs
```bash
docker logs $(docker ps -q --filter ancestor=csnli-api)
```

### Stop the Container
```bash
docker stop $(docker ps -q --filter ancestor=csnli-api)
```

### Restart the Container
```bash
docker start $(docker ps -a -q --filter ancestor=csnli-api)
```

## Testing the API

Once the container is running, you can test the API using curl:

```bash
curl -X POST "http://localhost:6000/csnli-lid" \
    -H "Content-Type: application/json" \
    -d '{"text": "i thght mosam dfrnt hoga bs fog h"}'
```

Expected output:
```json
{
    "csnli_op": {
        "text_str": "i thght mosam dfrnt hoga bs fog h",
        "text_tokenized": ["i", "thght", "mosam", "dfrnt", "hoga", "bs", "fog", "h"],
        "norm_text": ["i", "thought", "मौसम", "different", "होगा", "बस", "fog", "है"],
        "lid": ["en", "en", "hi", "en", "hi", "hi", "en", "hi"]
    }
}
```

The output shows:
- `text_str`: Original input text
- `text_tokenized`: Words split into tokens
- `norm_text`: Normalized text with proper spelling and script
- `lid`: Language identification tags (en=English, hi=Hindi)

## API Endpoints

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
        "text_str": "your text here",
        "text_tokenized": ["your", "text", "here"],
        "norm_text": ["normalized", "text", "here"],
        "lid": ["language", "tags", "here"]
    }
}
```


## Prerequisites

- Docker installed on your system
- Python 3.7 (for local development/testing)

## Building the Docker Image (Optional)

If you want to build the image locally instead of pulling from Docker Hub:

1. First, generate the requirements file from the conda environment:
```bash
python convert_conda_to_requirements.py
```

2. Build the Docker image:
```bash
docker build -t csnli-api .
```



## Troubleshooting

1. If you get an error about missing enchant library:
   - The Dockerfile already includes the necessary system dependencies
   - Rebuild the image if you haven't already

2. If the container exits immediately:
   - Check the logs using `docker logs`
   - Make sure all model files are present in the correct directories

3. If the API is not accessible:
   - Verify the container is running using `docker ps`
   - Check if port 6000 is not being used by another process
   - Ensure you're using the correct port in your API calls

## Directory Structure

- `csnli_api.py`: Main FastAPI application
- `three_step_decoding.py`: Core processing logic
- `lang_tagger.py`: Language identification module
- `lid_models/`: Language identification models
- `nmt_models/`: Neural machine translation models
- `lm/`: Language models
- `dicts/`: Dictionary files
- `csnli-models/`: Additional model files 