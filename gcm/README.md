# GCM Toolkit Docker Setup

This repository contains a Docker setup for the GCM (Grammar-based Code-Mixing) Toolkit, which provides API calls to utilize the aligner and code-mix generator functionalities in a simple manner.

## Quick Start

### Pull Docker Image
```bash
docker pull prakod/gcm-codemix-generator
```

### Run the Docker Container
```bash
docker run -p 5001:5000 -p 6001:6000 -d prakod/gcm-codemix-generator
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

The API service will now be accessible from your host machine - these APIs can now be accessed in your host.

- Ensure you are in the "library" folder

- To use the GCM APIs from the docker image, you can find examples in [ExampleNotebook.ipynb](../examples/ExampleNotebook.ipynb) notebook and run the cells to see the functionality of the API.

- You can change the ports but the example demostrations work with the default ports - if you change ports please edit the usage example suitably.


## Troubleshooting

1. If you can't access the Jupyter Notebook:
   - Make sure port 5000 is not being used by another process
   - Check if the container is running using `docker ps`

2. If the Flask API is not accessible:
   - Verify the container is running
   - Ensure port 6000 is available
   - Check if you're in the correct directory when running the Flask commands 

