
# About Repository :

For a MaskRCnn model trained with detectron2 running on localhost, it is to present the interface environment built with Gradio on docker, where the client can reach and make inferences on the predictions of the trained model.

<br><br>

## Docker build

```
docker build -t detectron2 -f Dockerfile .
```

## Run Docker

```
docker run -p 8888:8888 --gpus all -it detectron2
```

## Go To 
```
http://localhost:8888/
```