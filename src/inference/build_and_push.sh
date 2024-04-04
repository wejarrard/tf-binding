#!/usr/bin/env bash

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 016114370410.dkr.ecr.us-west-2.amazonaws.com
docker build -t tf-binding-inference .
docker tag tf-binding-inference:latest 016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference:latest
docker push 016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference:latest
python batch_transform.py
