# W_CellDetect_Stardist_HE
BIAFLOWS workflow for Cell/Nuclei detection, encapsulating Stardist Python code (https://github.com/mpicbg-csbd/stardist/) originally developed by Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers as published in Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

Applies a Stardist pre-trained model (versatile_fluo or versatile_HE) depending on the input images ie. uses both models for a dataset including both fluorescence (grayscale or RGB where all channels are equal) and H&E stained (RGB where channels are not equal) images.

This version uses tensorflow CPU version (See Dockerfile) to ensure compatibility with a larger number of computers. A GPU version should be possible by adapting the Dockerfile with tensorflow-gpu and/or nvidia-docker images.
