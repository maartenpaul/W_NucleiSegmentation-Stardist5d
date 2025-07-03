# W_CellSegmentation-SAM
BIAFLOWS workflow for cell segmentation, encapsulating StarDist Python code (https://github.com/mpicbg-csbd/stardist/) originally developed by Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers as published in Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

This workflow is modified from https://github.com/Neubias-WG5/W_NucleiSegmentation-Stardist to work with 5D images within BIOMERO.

## Features
- Supports multi-dimensional images (TZCYX format) with automatic dimension handling
- Applies StarDist pre-trained models (versatile_fluo or versatile_HE) based on input image characteristics
- Automatic tiling for large images with configurable tile sizes and overlap
- Processes selected channels, time points, and z-slices or all dimensions when set to -1
- Handles both fluorescence (grayscale or RGB where all channels are equal) and H&E stained (RGB where channels differ) images

## Processing Options
- **Auto-tiling**: Automatically uses block processing for images larger than specified tile sizes
- **Manual tiling**: Uses standard tiling approach for smaller images
- **Dimension selection**: Process specific or all channels, time points, and z-slices
- **Configurable parameters**: Probability thresholds, NMS thresholds, normalization percentiles, and scale factors

This version uses tensorflow CPU version (See Dockerfile) to ensure compatibility with a larger number of computers. A GPU version should be possible by adapting the Dockerfile with tensorflow-gpu and/or nvidia-docker images.

## Issues
For issues or questions, please use the GitHub issue tracker.
