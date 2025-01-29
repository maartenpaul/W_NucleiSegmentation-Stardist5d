# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.


import sys
import numpy as np
import os
import imageio
from tifffile import imread, imwrite
import skimage
import skimage.color
from csbdeep.utils import normalize
from stardist import random_label_cmap
from stardist.models import StarDist2D
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics

def run_startdist(bj,models,img):
    fluo = True
    n_channel = 3 if img.ndim == 3 else 1

    if n_channel == 3:
        if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,0], img[:,:,2]):
            img = skimage.color.rgb2gray(img)
        else:
            fluo = False
    axis_norm = (0,1)
    img = normalize(img, bj.parameters.stardist_norm_perc_low, bj.parameters.stardist_norm_perc_high,axis=axis_norm)
    if fluo:
        labels, details = models[0].predict_instances(img,
                                                       prob_thresh=bj.parameters.stardist_prob_t,
                                                       nms_thresh=bj.parameters.stardist_nms_t)
    else:
        labels, details = models[1].predict_instances(img,
                                                     prob_thresh=bj.parameters.stardist_prob_t,
                                                     nms_thresh=bj.parameters.stardist_nms_t)
    labels = labels.astype(np.uint16)
    bj.job.update(status=Job.RUNNING, progress=30, statusComment="Maxiumum value in labels: {}".format(np.max(labels)))
    return labels

def main(argv):
    base_path = "{}".format(os.getenv("HOME"))
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=False, **bj.flags)
        list_imgs = [image.filepath for image in in_imgs]
        nuc_channel = bj.parameters.nuc_channel
        channels = bj.parameters.channels
        time_series = bj.parameters.time_series
        z_slices = bj.parameters.z_slices

        # 2. Run Stardist model
        bj.job.update(progress=25, statusComment="Launching workflow...")
        bj.job.update(progress=30, statusComment="Processing images with channels is {}, time_series is {}, z_slices is {}".format(channels, time_series, z_slices))
        np.random.seed(17)
        model_fluo = StarDist2D(None, name='2D_versatile_fluo', basedir='/models/')
        model_he = StarDist2D(None, name='2D_versatile_he', basedir='/models/')
        models = [model_fluo, model_he]

        for img_path in list_imgs:
            img = imread(img_path)
            dims = img.shape
            
            if len(dims) == 5:
                _, nz, nt, nx, ny = img.shape
                labels = np.zeros((1,nz,nt,nx,ny), dtype=np.uint16)
                for z, t in np.ndindex(nz, nt):
                    labels[nuc_channel,z,t] = run_startdist(bj,models,img[nuc_channel,z,t])

            elif len(dims) == 4:
                if channels and z_slices:
                    _, nz, nx, ny = img.shape
                    labels = np.zeros((1,nz,nx,ny), dtype=np.uint16)
                    for z in range(nz):
                        labels[0,z] = run_startdist(bj,models,img[nuc_channel,z])
                        
                elif channels and time_series:
                    _, nt, nx, ny = img.shape
                    labels = np.zeros((1,nt,nx,ny), dtype=np.uint16)
                    for t in range(nt):
                        labels[0,t] = run_startdist(bj,models,img[nuc_channel,t])
                        
                elif z_slices and time_series:
                    nz, nt, nx, ny = img.shape
                    labels = np.zeros((nz,nt,nx,ny), dtype=np.uint16)
                    for z, t in np.ndindex(nz, nt):
                        labels[z,t] = run_startdist(bj,models,img[z,t])

            elif len(dims) == 3:
                if z_slices:
                    nz, nx, ny = img.shape
                    labels = np.zeros((nz,nx,ny), dtype=np.uint16)
                    for z in range(nz):
                        labels[z] = run_startdist(bj,models,img[z])
                        
                elif time_series:
                    nt, nx, ny = img.shape
                    labels = np.zeros((nt,nx,ny), dtype=np.uint16)
                    for t in range(nt):
                        labels[t] = run_startdist(bj,models,img[t])
                        
                elif channels:
                    labels = run_startdist(bj,models,img[nuc_channel])

            elif len(dims) == 2:
                labels = run_startdist(bj,models,img)

            bj.job.update(progress=90, statusComment=f"Objects detected in image: {np.any(labels>0)}")
            imwrite(os.path.join(out_path,os.path.basename(img_path)), labels)

        # 3. Upload data
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Finish
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
