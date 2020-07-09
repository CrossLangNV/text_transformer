FROM python:3.6-slim


# install the latest upgrades
RUN apt-get update && apt-get -y dist-upgrade

RUN pip install flair

COPY data /opt/data
COPY model /opt/model
COPY transform.py /opt/transform.py



WORKDIR /opt/
