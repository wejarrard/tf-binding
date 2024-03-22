aws ecr create-repository --repository-name tf-binding-inference --region us-west-2

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference

docker tag tf-binding-inference:latest 016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference:0.0.1

docker push 016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference:0.0.1
