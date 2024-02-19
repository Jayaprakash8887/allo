FROM continuumio/anaconda3:2023.03-1
WORKDIR /root
RUN apt-get update && apt-get install -y curl file
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=$PATH:/root/.cargo/bin \
    OPENAI_API_KEY=$OPENAI_API_KEY \
    LOG_LEVEL=$LOG_LEVEL \
    BHASHINI_ENDPOINT_URL=$BHASHINI_ENDPOINT_URL \
    BHASHINI_API_KEY=$BHASHINI_API_KEY \
    OCI_ENDPOINT_URL=$OCI_ENDPOINT_URL \
    OCI_REGION_NAME=$OCI_REGION_NAME \
    OCI_BUCKET_NAME=$OCI_BUCKET_NAME \
    OCI_SECRET_ACCESS_KEY=$OCI_SECRET_ACCESS_KEY \
    OCI_ACCESS_KEY_ID=$OCI_ACCESS_KEY_ID \
    SERVICE_ENVIRONMENT=$SERVICE_ENVIRONMENT
RUN apt-get update && apt install build-essential --fix-missing -y
RUN apt-get install ffmpeg -y
COPY requirements.txt /root/
RUN pip3 install -r requirements.txt
COPY ./main.py /root/
COPY ./cloud_storage_oci.py /root/
COPY ./io_processing.py /root/
COPY ./translator.py /root/
COPY ./logger.py /root/
COPY ./utils.py /root/
COPY ./config.ini /root/
COPY ./config_util.py /root/
EXPOSE 8022
COPY script.sh /root/
ENTRYPOINT ["bash","script.sh"]