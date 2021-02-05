FROM python:3.6-slim

# install the latest upgrades
RUN apt-get update && apt-get -y dist-upgrade

COPY transformer /opt/transformer
RUN pip install -r /opt/transformer/requirements.txt
COPY app.py /opt

WORKDIR /opt

CMD ["python", "app.py"]
