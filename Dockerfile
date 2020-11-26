FROM python:3.7-stretch

# -----------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.5.1 && pip install . && \
    rm -r /Cytomine-python-client
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Install BIAFLOWS-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/biaflows-utilities.git && \
    cd /biaflows-utilities/ && git checkout tags/v0.8.8 && pip install .

# install utilities binaries
RUN chmod +x /biaflows-utilities/bin/*
RUN cp /biaflows-utilities/bin/* /usr/bin/ && \
    rm -r /biaflows-utilities

# -----------------------------------------------------------------------------
# Install Stardist and tensorflow
RUN pip uninstall -y numba
RUN pip install tensorflow==1.15
RUN pip install stardist==0.5.0
RUN mkdir -p /models && \
    cd /models && \
    mkdir -p 2D_versatile_fluo
ADD config.json /models/2D_versatile_fluo/config.json
ADD thresholds.json /models/2D_versatile_fluo/thresholds.json
ADD weights_last.h5 /models/2D_versatile_fluo/weights_best.h5
RUN chmod 444 /models/2D_versatile_fluo/config.json
RUN chmod 444 /models/2D_versatile_fluo/thresholds.json
RUN chmod 444 /models/2D_versatile_fluo/weights_best.h5


# -----------------------------------------------------------------------------
# Install scripts
ADD descriptor.json /app/descriptor.json
RUN mkdir -p /app
ADD run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]

