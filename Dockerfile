FROM ubuntu:latest
LABEL authors="giorg"

ENTRYPOINT ["top", "-b"]