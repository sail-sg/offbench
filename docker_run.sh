TAG="offbench:latest"
docker build --build-arg uid=$(id -u ${USER}) --build-arg user=${USER} -t ${TAG} .
docker run --gpus=all --net=host --privileged \
    -v `pwd`:/offbench \
    -it ${TAG} \
    /bin/bash
