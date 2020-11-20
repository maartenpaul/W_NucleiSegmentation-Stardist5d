# W_CellDetect_Stardist_HE
BIAFLOWS workflow for Cell/Nuclei detection, encapsulating Stardist Python code (https://github.com/mpicbg-csbd/stardist/) originally developed by Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers as published in Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

It applies a Stardist pre-trained model (versatile_HE) to input images in BIAFLOWS platform.

This version uses tensorflow CPU version (See Dockerfile) to ensure compatibility with a larger number of computers. A GPU version should be possible by adapting the Dockerfile with tensorflow-gpu and/or nvidia-docker images.
