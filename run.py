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
import skimage
import skimage.color
from csbdeep.utils import normalize
from stardist import random_label_cmap
from stardist.models import StarDist2D
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics

def run_startdist(img):
    fluo = True
    n_channel = 3 if img.ndim == 3 else 1

    if n_channel == 3:
        # Check if 3-channel grayscale image or actually an RGB image
        if np.array_equal(img[:,:,0],img[:,:,1]) and np.array_equal(img[:,:,0],img[:,:,2]):
            img = skimage.color.rgb2gray(img)
        else:
            fluo = False

    # normalize channels independently (0,1,2) normalize channels jointly (0,1)
    axis_norm = (0,1)
    img = normalize(img, bj.parameters.stardist_norm_perc_low, bj.parameters.stardist_norm_perc_high, axis=axis_norm)

    #Stardist model prediction with thresholds
    if fluo:
        labels, details = model_fluo.predict_instances(img,
                                                        prob_thresh=bj.parameters.stardist_prob_t,
                                                        nms_thresh=bj.parameters.stardist_nms_t) 
    else:
        labels, details = model_he.predict_instances(img,
                                                        prob_thresh=bj.parameters.stardist_prob_t,
                                                        nms_thresh=bj.parameters.stardist_nms_t)
    
    # Convert labels to uint16 for BIAFLOWS
    labels = labels.astype(np.uint16)
    return labels

def main(argv):
    base_path = "{}".format(os.getenv("HOME"))
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=True, **bj.flags)
        list_imgs = [image.filepath for image in in_imgs]
        nuc_channel = bj.parameters.nuc_channel

        # 2. Run Stardist model on input images
        bj.job.update(progress=25, statusComment="Launching workflow...")
        
        #Loading pre-trained Stardist model
        np.random.seed(17)

        lbl_cmap = random_label_cmap()
        model_fluo = StarDist2D(None, name='2D_versatile_fluo', basedir='/models/')
        model_he = StarDist2D(None, name='2D_versatile_he', basedir='/models/')

        #Go over images
        for img_path in list_imgs:
            #check if image is 2D or 3D
            img = imageio.imread(img_path)
            nz, _, nt = img.shape[:3]
            if img.ndim == 5:
                processed_img = np.zeros_like(img)
                for z, t in np.ndindex(nz, nt):
                    # Process single xy slice
                    processed_slice = run_startdist(img[z,nuc_channel,t])
                    # Store processed slice
                    processed_img[z,nuc_channel,t] = processed_slice
            else:
                labels = run_startdist(img)

            imageio.imwrite(os.path.join(out_path,os.path.basename(img_path)), labels)

        # 3. Upload data to BIAFLOWS
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Pipeline finished
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
