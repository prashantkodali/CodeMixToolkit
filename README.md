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
