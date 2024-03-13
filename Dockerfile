# BUILDING: docker build -t <container_name> .
# RUNNING: docker run <container_name> <python_program_path> <config_file_path>
# e.g. docker run --network="host" anomaly_detection
FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev
COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip3 install -r requirements.txt
COPY . /
# CMD ["python3", "main.py", "-c", "deployment/NAIADES_Braila.json"]
# CMD ["python3", "main.py", "-c", "deployment/NAIADES_Braila_flow.json"]
CMD ["python3", "main.py", "-c", "FileConfig.json"]
