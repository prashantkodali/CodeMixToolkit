# GCM Toolkit Docker Setup

This repository contains a Docker setup for the GCM (Grammar-based Code-Mixing) Toolkit, which provides API calls to utilize the aligner and code-mix generator functionalities in a simple manner.

## Quick Start

### Pull Docker Image
```bash
docker pull prakod/gcm-codemix-generator
```

### Run the Docker Container
```bash
docker run -p 5000:5000 -p 6000:6000 -d prakod/codemix-gcm-generator
```

This will create a container based on the Docker image. Get the ID of the container using:
```bash
docker ps
```

### Access the Container
```bash
docker exec -it <container_id> bash
```

Once inside the container:
```bash
conda activate gcm-venv
git clone https://github.com/prashantkodali/CodeMixToolkit.git
```

## Running Services

### Jupyter Notebook
```bash
jupyter notebook --ip 0.0.0.0 --port 5000 --no-browser --allow-root
```

### Flask API
1. Navigate to the "library" folder
2. Run the following commands:
```bash
export FLASK_APP=gcmgenerator
flask run -h 0.0.0.0 -p 6000
```

The API service will now be accessible from your host machine.

## Example Usage

To use the GCM APIs from the docker image, you can find examples in the [GCM Generator API Demo.ipynb](../GCM Generator API Demo.ipynb) notebook. Run the cells to see the functionality of the API.

## Ports
- Port 5000: Jupyter Notebook
- Port 6000: Flask API

## Troubleshooting

1. If you can't access the Jupyter Notebook:
   - Make sure port 5000 is not being used by another process
   - Check if the container is running using `docker ps`

2. If the Flask API is not accessible:
   - Verify the container is running
   - Ensure port 6000 is available
   - Check if you're in the correct directory when running the Flask commands 



   ## CODE-MIX GENERATOR - MODIFIED DOCKER IMAGE

- This modified docker image contains API calls to utilise the aligner and codemix-generator functionalities in a simple manner.

### Pull docker image

- Link to docker hub: https://hub.docker.com/r/prakod/gcm-codemix-generator
- Alternatively, use the command 
```
docker pull prakod/gcm-codemix-generator
```

### Instructions to run the docker image (after pulling docker image)
```
docker run -p 5000:5000 -p 6000:6000 -d prakod/codemix-gcm-generator (this can alternatively be done using Docker desktop)
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

- This command runs the API service in the docker image - these APIs can now be accessed in your host.

- To use the GCM APIs from the docker image, you can find examples in [GCM Generator API Demo.ipynb](GCM Generator API Demo.ipynb) notebook and run the cells to see the functionality of the API.
