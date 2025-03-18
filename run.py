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
from tifffile import imwrite, TiffFile
import skimage
import skimage.color
from csbdeep.utils import normalize
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist import random_label_cmap
from stardist.models import StarDist2D
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics

def convert_to_5d_from_tifffile(volume, axes, target="XYZCT"):
    """
    Convert a numpy array from TiffFile to 5D dimensions suitable for OMERO
    
    Parameters
    ----------
    volume : numpy.ndarray
        Image data from tifffile's asarray()
    axes : str
        Axes string from tifffile (e.g., 'TZCYX', 'YX', etc.)
    target : str, optional
        String specifying the desired dimension order, default is "XYZCT"
        
    Returns
    -------
    img_5d : numpy.ndarray or tuple
        5D numpy array with dimensions ordered according to target
        When unpacked as a tuple, returns (img_5d, target)
    """
    # Validate input volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input volume must be a numpy.ndarray")
    
    # Standardize to uppercase
    axes = axes.upper()
    target = target.upper()
    
    # Validate axes dimensions match array dimensions
    if len(axes) != volume.ndim:
        raise ValueError(f"Axes string '{axes}' does not match array dimensions {volume.ndim}")
    
    # Some TIFF files use 'S' for samples/channels, convert to 'C' for consistency
    axes = axes.replace('S', 'C')
        
    # Validate target dimensions
    if len(target) != 5:
        raise ValueError(f"Target dimensions must have exactly 5 dimensions, got '{target}'")
    
    if set(target) != set("XYZCT"):
        raise ValueError("Target dimensions must contain letters X, Y, Z, C, and T exactly once")
    
    # Create a 5D array by adding missing dimensions
    img_5d = volume
    current_order = axes
    
    # Add missing dimensions
    for dim in "XYZCT":
        if dim not in current_order:
            img_5d = np.expand_dims(img_5d, axis=-1)
            current_order += dim
    
    # Reorder dimensions if needed
    if current_order != target:
        # Create list of current positions for each dimension
        current_positions = []
        for dim in target:
            current_positions.append(current_order.index(dim))
        
        # Rearrange dimensions
        img_5d = np.moveaxis(img_5d, current_positions, range(len(target)))
    
    # Return both the array and target, allowing for flexible unpacking
    class ReturnValue(tuple):
        """Custom return class to allow both direct access and unpacking"""
        def __new__(cls, img, axes):
            return tuple.__new__(cls, (img, axes))
            
        def __repr__(self):
            return repr(self[0])
            
        # Make the first element (the image) accessible directly
        def __array__(self, dtype=None):
            return np.asarray(self[0], dtype=dtype)
    
    return ReturnValue(img_5d, target)

def should_use_big_model(img_shape, params):
    """
    Determine whether to use predict_instances_big based on image size and parameters
    
    Args:
        img_shape: Shape of the 2D image slice (Y, X)
        params: Parameters object with tile_size_x, tile_size_y, and auto_tiling attributes
        
    Returns:
        Boolean indicating whether to use predict_instances_big
    """
    # If auto_tiling is disabled, use the explicit big_image parameter
    if not params.auto_tiling:
        return params.big_image
    
    # Get image dimensions
    height, width = img_shape
    
    # Check if image is larger than specified tile sizes
    return height > params.tile_size_y or width > params.tile_size_x

def calculate_n_tiles(img_shape, tile_size_y, tile_size_x):
    """
    Calculate appropriate n_tiles parameter based on image dimensions and tile sizes
    
    Args:
        img_shape: Shape of the 2D image slice (Y, X)
        tile_size_y: Tile size in Y dimension
        tile_size_x: Tile size in X dimension
        
    Returns:
        Tuple (n_tiles_y, n_tiles_x) for StarDist predict_instances
    """
    # Get image dimensions
    height, width = img_shape
    
    # Calculate number of tiles needed in each dimension
    n_tiles_y = max(1, int(np.ceil(height / tile_size_y)))
    n_tiles_x = max(1, int(np.ceil(width / tile_size_x)))
    
    return (n_tiles_y, n_tiles_x)

def create_normalizer(img_slice, params):
    """
    Create a custom normalizer for StarDist
    
    Args:
        img_slice: 2D image slice
        params: Parameters with normalization settings
        
    Returns:
        Normalizer object for StarDist prediction functions
    """
    class MyNormalizer(Normalizer):
        def __init__(self, mi, ma):
            self.mi, self.ma = mi, ma
            
        def before(self, x, axes):
            return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
            
        def after(*args, **kwargs):
            assert False
            
        @property
        def do_after(self):
            return False
    
    # Calculate percentiles for normalization
    mi, ma = np.percentile(img_slice, [params.stardist_norm_perc_low, 
                                      params.stardist_norm_perc_high])
    
    return MyNormalizer(mi, ma)

def run_stardist_on_slice(img_slice, models, params):
    """
    Run StarDist on a 2D image slice with automatic handling of image size and scaling
    
    Args:
        img_slice: 2D image slice
        models: StarDist models (fluo and HE)
        params: Parameters object with StarDist settings
        
    Returns:
        Segmentation mask for the slice
    """
    # Determine if fluorescence or H&E model should be used
    fluo = True
    n_channel = 3 if img_slice.ndim == 3 else 1
    
    # Convert RGB to grayscale if all channels are identical
    if n_channel == 3:
        if np.array_equal(img_slice[:,:,0], img_slice[:,:,1]) and np.array_equal(img_slice[:,:,0], img_slice[:,:,2]):
            img_slice = skimage.color.rgb2gray(img_slice)
        else:
            fluo = False
    
    # Select the appropriate model
    model = models[0] if fluo else models[1]
    
    # Determine if we should use predict_instances_big based on image size
    use_big = should_use_big_model(img_slice.shape, params)
    
    # Get scale factor
    scale = params.scale_factor
    
    if use_big:
        # Create normalizer for large images
        normalizer = create_normalizer(img_slice, params)
        
        # Set up block processing parameters
        block_size = (params.tile_size_y, params.tile_size_x)
        
        # Ensure block_overlap is not too large compared to block_size
        min_block_dimension = min(block_size)
        block_overlap = min(params.block_overlap, min_block_dimension // 4)
        
        # Use same value for context as overlap by default
        context = block_overlap
        
        # Process with predict_instances_big
        labels, polys = model.predict_instances_big(
            img_slice, 
            axes='YX', 
            block_size=block_size, 
            min_overlap=block_overlap, 
            context=context,
            normalizer=normalizer,
            scale=scale,
            prob_thresh=params.stardist_prob_t,
            nms_thresh=params.stardist_nms_t
        )
    else:
        # Calculate n_tiles for regular processing
        n_tiles = calculate_n_tiles(img_slice.shape, params.tile_size_y, params.tile_size_x)
        
        # Normalize image directly for smaller images
        img_slice_norm = normalize(img_slice, 
                            params.stardist_norm_perc_low, 
                            params.stardist_norm_perc_high,
                            axis=(0,1))
        
        # Process with standard predict_instances
        labels, polys = model.predict_instances(
            img_slice_norm,
            prob_thresh=params.stardist_prob_t,
            nms_thresh=params.stardist_nms_t,
            n_tiles=n_tiles,
            scale=scale
        )
    
    return labels.astype(np.uint16)

def process_image(img_path, models, bj, params):
    """
    Process image using convert_to_5d_from_tifffile to standardize dimensions.
    Converts input to TZCYX format and maintains this format for output.
    
    Args:
        img_path: Path to the input image
        models: StarDist models (fluo and HE)
        bj: BiaFlows job object
        params: Parameter object with processing settings
        
    Returns:
        tuple: (segmentation_labels, axes_string)
            - segmentation_labels: 5D array in TZCYX format
            - axes_string: Always "TZCYX"
    
    Notes:
        In TZCYX format:
        - img_5d.shape[0]: T dimension (time)
        - img_5d.shape[1]: Z dimension (depth/slices)
        - img_5d.shape[2]: C dimension (channels)
        - img_5d.shape[3]: Y dimension (height)
        - img_5d.shape[4]: X dimension (width)
    """
    with TiffFile(img_path) as tif:
        # Load image data and get axes information
        volume = tif.asarray()
        original_axes = tif.series[0].axes
        bj.job.update(status=Job.RUNNING, progress=30,
                     statusComment=f"Processing image with original axes: {original_axes}")
        
        # Convert to 5D with TZCYX order for processing
        bj.job.update(status=Job.RUNNING, progress=35,
                     statusComment="Converting to 5D TZCYX format...")
        img_5d, axes_5d = convert_to_5d_from_tifffile(volume, original_axes, target="TZCYX")
        
        # Track dimension sizes for clarity
        dims = {
            'T': img_5d.shape[0],  # Time
            'Z': img_5d.shape[1],  # Depth/slices
            'C': img_5d.shape[2],  # Channels
            'Y': img_5d.shape[3],  # Height
            'X': img_5d.shape[4]   # Width
        }
        bj.job.update(status=Job.RUNNING, progress=40,
                     statusComment=f"Dimensions (TZCYX): {dims}")
        
        # Get channel, time, and z-slice parameters
        nuc_channel = params.nuc_channel
        time_point = params.time_series
        z_slice = params.z_slices
        
        # Prepare channel list to process
        if isinstance(nuc_channel, (int, np.integer)):
            if nuc_channel == -1:
                # Process all channels
                channels_to_process = list(range(dims['C']))
            else:
                channels_to_process = [nuc_channel]
        elif isinstance(nuc_channel, (list, tuple, np.ndarray)):
            channels_to_process = list(nuc_channel)
        else:
            raise ValueError(f"Invalid nuc_channel type: {type(nuc_channel)}. Must be int or list of ints.")
        
        # Prepare time points to process
        if isinstance(time_point, (int, np.integer)):
            if time_point == -1:
                # Process all time points
                time_points_to_process = list(range(dims['T']))
            else:
                time_points_to_process = [time_point]
        elif isinstance(time_point, (list, tuple, np.ndarray)):
            time_points_to_process = list(time_point)
        else:
            raise ValueError(f"Invalid time_point type: {type(time_point)}. Must be int or list of ints.")
        
        # Prepare z-slices to process
        if isinstance(z_slice, (int, np.integer)):
            if z_slice == -1:
                # Process all z-slices
                z_slices_to_process = list(range(dims['Z']))
            else:
                z_slices_to_process = [z_slice]
        elif isinstance(z_slice, (list, tuple, np.ndarray)):
            z_slices_to_process = list(z_slice)
        else:
            raise ValueError(f"Invalid z_slice type: {type(z_slice)}. Must be int or list of ints.")
        
        # Validate indices
        for channel in channels_to_process:
            if channel >= dims['C']:
                raise ValueError(f"Invalid channel index {channel}. Image has {dims['C']} channels (0-{dims['C']-1})")
        
        for t in time_points_to_process:
            if t >= dims['T']:
                raise ValueError(f"Invalid time point {t}. Image has {dims['T']} time points (0-{dims['T']-1})")
        
        for z in z_slices_to_process:
            if z >= dims['Z']:
                raise ValueError(f"Invalid z-slice {z}. Image has {dims['Z']} z-slices (0-{dims['Z']-1})")
        
        # Create output array - shape matches the selected dimensions
        output_shape = (
            len(time_points_to_process),
            len(z_slices_to_process),
            len(channels_to_process),
            dims['Y'],
            dims['X']
        )
        output = np.zeros(output_shape, dtype=np.uint16)
        
        # Calculate total slices for progress tracking
        total_slices = len(channels_to_process) * len(z_slices_to_process) * len(time_points_to_process)
        current_slice = 0
        
        # Process selected dimensions
        for ch_idx, channel in enumerate(channels_to_process):
            bj.job.update(status=Job.RUNNING,
                         statusComment=f"Processing channel {channel} ({ch_idx+1}/{len(channels_to_process)})")
            
            for t_idx, t in enumerate(time_points_to_process):
                for z_idx, z in enumerate(z_slices_to_process):
                    # Extract YX slice from current TZC position
                    img_slice = img_5d[t, z, channel, :, :]  # Note TZCYX order
                    
                    # Process the slice with new function
                    labels = run_stardist_on_slice(img_slice, models, params)
                    
                    # Store results in corresponding position
                    output[t_idx, z_idx, ch_idx, :, :] = labels
                    
                    # Update progress
                    current_slice += 1
                    progress = 40 + (50 * current_slice / total_slices)
                    bj.job.update(status=Job.RUNNING, progress=int(progress),
                                statusComment=f"Processed channel {channel}, T {t+1}/{dims['T']}, Z {z+1}/{dims['Z']}")
        
        # Add dimension mapping information to output metadata
        dimension_mapping = {
            'T': time_points_to_process,
            'Z': z_slices_to_process,
            'C': channels_to_process
        }
        
        # Keep TZCYX format for output, but with metadata about which slices were processed
        bj.job.update(status=Job.RUNNING, progress=90,
                     statusComment=f"Maintaining TZCYX format with processed dimensions: {dimension_mapping}")
        
        return output, "TZCYX", dimension_mapping

def main(argv):
    base_path = "{}".format(os.getenv("HOME"))
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")
        
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=False, **bj.flags)
        list_imgs = [image.filepath for image in in_imgs]
        
        # 2. Initialize StarDist models
        bj.job.update(progress=15, statusComment="Loading StarDist models...")
        np.random.seed(17)
        model_fluo = StarDist2D(None, name='2D_versatile_fluo', basedir='/models/')
        model_he = StarDist2D(None, name='2D_versatile_he', basedir='/models/')
        models = [model_fluo, model_he]
        
        # Process each image
        bj.job.update(progress=20, statusComment=f"Number of images: {len(list_imgs)}")
        for img_index, img_path in enumerate(list_imgs):
            img_name = os.path.basename(img_path)
            bj.job.update(progress=int(25 + (50 * img_index / len(list_imgs))),
                        statusComment=f"Processing image {img_index+1}/{len(list_imgs)}: {img_name}")
            
            try:
                # Process image using our updated function
                labels, axes, dimension_mapping = process_image(img_path, models, bj, bj.parameters)
                
                # Create output path
                output_path = os.path.join(out_path, img_name)
                
                # Save in TZCYX format
                metadata = {
                    'axes': axes,
                    'dimension_mapping': str(dimension_mapping),
                    'stardist_params': {
                        'prob_threshold': bj.parameters.stardist_prob_t,
                        'nms_threshold': bj.parameters.stardist_nms_t,
                        'scale_factor': bj.parameters.scale_factor
                    }
                }
                
                imwrite(output_path, labels,
                       metadata=metadata,
                       photometric='minisblack',
                       ome=True,
                       description='Processed with StarDist, standardized to TZCYX format')
                
                # Log information about the segmentation
                num_objects = len(np.unique(labels)) - 1  # Subtract 1 for background
                bj.job.update(progress=int(75 + (15 * (img_index + 1) / len(list_imgs))),
                            statusComment=f"Completed {img_name}: {num_objects} objects detected")
            
            except Exception as e:
                error_msg = f"Error processing {img_name}: {str(e)}"
                bj.job.update(progress=75, statusComment=error_msg)
                import traceback
                traceback.print_exc()
                # Continue with next image
                continue

        # 3. Upload data
        bj.job.update(progress=90, statusComment="Uploading results...")
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 90, "end": 95, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute metrics
        bj.job.update(progress=95, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Finish
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])