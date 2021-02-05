# COMPRISE - Text Transformation

This repository gives access to the tools that have been used, implemented and/or delivered to remove sensitive information from text, as part of the COMPRISE project.

- The ["Transformer"](./transformer) provides a script to apply this transformation. Documentation and installation instructions are in the [README](./transformer/README.md).

- The ["Model builder"](./model_builder) provides scripts to train a model. Documentation and installation instructions are in the [README](./model_builder/README.md).


The transformer can also be run as a RESTful service using docker with the instruction:
```
docker run -d -p 5000:5000 --name comprise-tt registry.gitlab.inria.fr/comprise/text-transformer
``` 
*Note: depending on your docker installation, you may have to run docker with sudo privileges*

The service is then listening on the port 5000 and expose the endpoint: `http://localhost:5000/transform`

Here is an example of a call to this service in python:

```
import requests
import json

text_to_transform= 'I live in Berlin'
ENDPOINT_URL = 'http://localhost:5000/transform'  # replace localhost with the proper hostname

# Transformation parameters
params = {'r': 'WORD'}

# Call the service
response = requests.post(ENDPOINT_URL, data=text_to_transform, params=json.dumps(params))

# Get the result
print(response.text)  #  'I live in Sweden'
```

Stop and start the service with:
```
docker stop comprise-tt
docker start comprise-tt
```
 
You can also use your locally stored models by mounting your models folder to the docker container:
```
docker run -d \
    -p 5000:5000 \
    -v "$(pwd)"/transformer/io/model:/opt/transformer/io/model \
    --name comprise-tt \
    registry.gitlab.inria.fr/comprise/text-transformer
``` 
 

If you use this tool, please cite:
```
@inproceedings{adelani:hal-02907939,
  TITLE = {{Privacy guarantees for de-identifying text transformations}},
  AUTHOR = {Adelani, David Ifeoluwa and Davody, Ali and Kleinbauer, Thomas and Klakow, Dietrich},
  URL = {https://hal.inria.fr/hal-02907939},
  BOOKTITLE = {{INTERSPEECH 2020}},
  ADDRESS = {Shanghai, China},
  YEAR = {2020},
  MONTH = Oct,
  KEYWORDS = {Differential privacy ; Spoken language understanding ; Named entity recognition ; Intent detection},
  PDF = {https://hal.inria.fr/hal-02907939/file/adelani_IS20.pdf},
  HAL_ID = {hal-02907939},
  HAL_VERSION = {v1},
}
```
