FROM python:3.7-bullseye

# -----------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip install . && \
    rm -r /Cytomine-python-client
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Install BIAFLOWS-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/biaflows-utilities.git && \
    cd /biaflows-utilities/ && git checkout tags/v0.9.2 && pip install .

# install utilities binaries
RUN chmod +x /biaflows-utilities/bin/*
RUN cp /biaflows-utilities/bin/* /usr/bin/ && \
    rm -r /biaflows-utilities

# -----------------------------------------------------------------------------
RUN pip install imageio

# Install Stardist and tensorflow
RUN pip install tensorflow==1.15
RUN pip install keras==2.1.6
RUN pip3 install h5py==2.10.0
RUN pip install protobuf==3.20.*
RUN pip install stardist[tf1]==0.5.0
RUN pip install -U tifffile[all]

RUN mkdir -p /models && \
    cd /models && \
    mkdir -p 2D_versatile_fluo && \
    mkdir -p 2D_versatile_he

# Add fluo model
ADD fluo_config.json /models/2D_versatile_fluo/config.json
ADD fluo_thresholds.json /models/2D_versatile_fluo/thresholds.json
ADD fluo_weights.h5 /models/2D_versatile_fluo/weights_best.h5
RUN chmod 444 /models/2D_versatile_fluo/config.json
RUN chmod 444 /models/2D_versatile_fluo/thresholds.json
RUN chmod 444 /models/2D_versatile_fluo/weights_best.h5
# Add HE model
ADD he_config.json /models/2D_versatile_he/config.json
ADD he_thresholds.json /models/2D_versatile_he/thresholds.json
ADD he_weights.h5 /models/2D_versatile_he/weights_best.h5
RUN chmod 444 /models/2D_versatile_he/config.json
RUN chmod 444 /models/2D_versatile_he/thresholds.json
RUN chmod 444 /models/2D_versatile_he/weights_best.h5

# -----------------------------------------------------------------------------
# Install scripts
ADD descriptor.json /app/descriptor.json
RUN mkdir -p /app
ADD run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]
