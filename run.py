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


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import os
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
import imageio
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap
from stardist.models import StarDist2D
from cytomine import cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job
from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics
from PIL import Image
import argparse
import json
import logging


__author__ = "Maree Raphael <raphael.maree@uliege.be>"

def main(argv):
    base_path = "{}".format(os.getenv("HOME"))
    problem_cls = CLASS_OBJSEG

    with NeubiasJob.from_cli(argv) as nj:
        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, is_2d=True, **nj.flags)
        list_imgs = [image.filepath for image in in_imgs]
        
        # 2. Run Stardist model on input images
        nj.job.update(progress=25, statusComment="Launching workflow...")
        
        #Loading pre-trained Stardist model
        np.random.seed(17)
        lbl_cmap = random_label_cmap()
        #Stardist H&E model downloaded from https://github.com/mpicbg-csbd/stardist/issues/46
        #Stardist H&E model downloaded from https://drive.switch.ch/index.php/s/LTYaIud7w6lCyuI
        model = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')   #use local model file in ~/models/2D_versatile_HE/

        #Go over images
        for img_path in list_imgs:
            img = imageio.imread(img_path)
            n_channel = 3 if img.ndim == 3 else img.shape[-1]
            # normalize channels independently (0,1,2) normalize channels jointly
            axis_norm = (0,1)
            img = normalize(img, nj.parameters.stardist_norm_perc_low, nj.parameters.stardist_norm_perc_high, axis=axis_norm)

            #Stardist model prediction with thresholds
            labels, details = model.predict_instances(img,
                                                      prob_thresh=nj.parameters.stardist_prob_t,
                                                      nms_thresh=nj.parameters.stardist_nms_t)
            imageio.imwrite(os.path.join(out_path,os.path.basename(img_path)), labels)

        # 3. Upload data to BIAFLOWS
        upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

        # 5. Pipeline finished
        nj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
