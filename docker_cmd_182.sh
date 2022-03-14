#docker_cmd_93.sh
img="pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime"
#nvcr.io/nvidia/pytorch:18.01-py3



docker run --gpus all  --privileged=true   --workdir /git --name "ssast"  -e DISPLAY --ipc=host -d --rm  -p 6625:4452  \
-v /raid/git/ssast:/git/ssast \
 -v  /raid/git/datasets:/git/datasets \
 $img sleep infinity


docker exec -it ssast /bin/bash


#docker images |grep pytorch  | grep  "1.9.0"

