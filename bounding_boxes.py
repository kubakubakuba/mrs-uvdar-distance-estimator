from metavision_core.event_io import RawReader
from metavision_ml.preprocessing import histo

from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_opening, gaussian_filter

import numpy as np

def find_bounding_boxes(raw, delta_t, start_ts, height, width, bounding_box_size=20):

	events = raw.load_delta_t(delta_t)
	events['t'] -= int(start_ts)  #adjust timestamps to be relative

	# create an event count map (histogram)
	tbins = 1
	volume = np.zeros((tbins, 2, height, width), dtype=np.float32)
	histo(events, volume, delta_t)

	#get the number of events per pixel
	event_count_map = volume[0].sum(axis=0)

	#gaussian filter to get rid of the noise
	event_count_map_smoothed = gaussian_filter(event_count_map, sigma=1)

	#thresholding
	threshold_value = threshold_otsu(event_count_map_smoothed)
	binary_map = event_count_map_smoothed > threshold_value

	#apply morphological operations to clean up noise
	binary_map = binary_opening(binary_map, structure=np.ones((3, 3)))

	#labeling connected components
	labeled_map = label(binary_map)

	#extract properties of connected components
	regions = regionprops(labeled_map)

	#filter regions and extract bounding boxes
	bounding_boxes = []
	for region in regions:
		#extract the bounding box coordinates
		minr, minc, maxr, maxc = region.bbox
		bbox_height = maxr - minr
		bbox_width = maxc - minc

		#filter out small regions (noise)
		min_size = 5
		if bbox_height >= min_size and bbox_width >= min_size:
			#inflate the bounding box to a fit center
			centroid_r, centroid_c = region.centroid
			half_size = bounding_box_size / 2
			minr_adj = max(int(centroid_r) - half_size, 0)
			maxr_adj = min(int(centroid_r) + half_size, height)
			minc_adj = max(int(centroid_c) - half_size, 0)
			maxc_adj = min(int(centroid_c) + half_size, width)

			bounding_boxes.append((minc_adj, minr_adj, maxc_adj, maxr_adj))
	
	return (bounding_boxes, event_count_map_smoothed)