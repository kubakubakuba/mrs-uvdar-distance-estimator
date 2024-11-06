from metavision_core.event_io import RawReader, EventsIterator
from metavision_sdk_analytics import DominantFrequencyEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_ml.preprocessing import histo, viz_histo

from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_opening, gaussian_filter
from scipy.signal import find_peaks

import numpy as np

event_dtype = np.dtype([('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])

def cut_timelt_events(events, time):
	"""
		Cut out events that are under some time window.

		Args:
			events: An array of event from an event camera in format (x, y, p, t).
			time (int): A time below which all event will be removed.

		Returns:
			Filtered events that fall in this condition.
	"""

	return events[events['t'] > time]

def filter_events(events, areas):
	"""
	Filters events in multiple areas, each defined by a position (x, y) and size (width, height).
	
	Args:
		events: An array of events from an event camera.
		areas: A list of tuples, where each tuple contains:
			- min_row (int): Top-left row coordinate of the bounding box.
			- min_col (int): Top-left column coordinate of the bounding box.
			- max_row (int): Bottom-right row coordinate of the bounding box.
			- max_col (int): Bottom-right column coordinate of the bounding box.
	
	Returns:
		Filtered events that fall within any of the specified areas.
	"""

	filtered_events = []

	for min_r, min_c, max_r, max_c in areas:
		x = min_r
		y = min_c
		width = max_c - min_c
		height = max_r - min_r
		
		filtered_events.append(events[(events['x'] >= x) & (events['x'] < x + width) & (events['y'] >= y) & (events['y'] < y + height)])
	
	return filtered_events

def polarize_events(events, polarity = -1):
	"""
	Polarizes events.
	
	Args:
		events: An array of events from an event camera.
		polarity: The polarity to filter events with. Can be either 0 or 1. If set to -1 or ony other value it returns all the events.
	
	Returns:
		Filtered events that have the specified polarity.
	"""

	return events[events['p'] != polarity]

def filter_events_singlebox(events, area):
	"""
	Filters events in one area.
	
	Args:
		events: An array of events from an event camera.
		areas_sizes: A tuple, which contains:
			- min_row (int): Top-left row coordinate of the bounding box.
			- min_col (int): Top-left column coordinate of the bounding box.
			- max_row (int): Bottom-right row coordinate of the bounding box.
			- max_col (int): Bottom-right column coordinate of the bounding box.
	
	Returns:
		Filtered events that fall within any of the specified areas.
	"""

	min_r, min_c, max_r, max_c = area
	
	return events[(events['x'] >= min_c) & (events['x'] < max_c) & (events['y'] >= min_r) & (events['y'] < max_r)]

def filter_raws(raws_boxes):
	"""
	Filters a raw recording with a bounding box.
	
	Args:
		raws_boxes: An array of tuples of (RawReader, [(min_r, min_c, max_r, max_c), ...])
	
	Returns:
		An array of filtered events for input recordings.
	"""

	filtered = []
	#print(raws_boxes[0][1])
	for (raw, boxes) in raws_boxes:
		raw.reset()  # Ensure we're at the beginning of the recording
		events = raw.load_n_events(-1)  # load all events
		events = filter_events(events, boxes)

		filtered.append(events)

	return filtered

def filter_raw(raws_boxes):
	filtered = []

	raw, boxes = raws_boxes

	raw.reset()
	events = raw.load_n_events(-1)
	events = filter_events(events, boxes)

	filtered.append(events)

	return filtered

def move_events(events, offset=(0, 0)):
	"""
		Moves events to locations near (0, 0)

		Args:
			events: An array of events (x, y, p, t)

		Returns:
			events: An array of moved events (x, y, p, t)
	"""

	xo, yo = offset

	events = np.array(events)

	min_x = np.min(events['x'])
	min_y = np.min(events['y'])

	events['x'] -= min_x - xo
	events['y'] -= min_y - yo

	return events

def find_bounding_boxes(raw: RawReader, delta_t = 10000, start_ts = 0.1 * 1e6, bounding_box_size=20):
	"""
	Processes a raw event recording and locates areas of interest (such as LEDs) in the event stream by identifying bounding boxes.
	
	Args:
		raw: A RawReader object containing the event stream data.
		delta_t: The time interval (in microseconds) over which to accumulate events for each frame. Default is 500 microseconds.
		start_ts: The starting timestamp (in microseconds) from which to begin processing the event stream. Default is 0.5 seconds.
		bounding_box_size: The approximate size of the bounding boxes to identify (in pixels). Default is 20 pixels.
	
	Returns:
		tuple: A tuple containing two elements:
			- list of tuples: Each tuple represents a bounding box around an area of interest, containing:
				- min_row (int): Top-left row coordinate of the bounding box.
				- min_col (int): Top-left column coordinate of the bounding box.
				- max_row (int): Bottom-right row coordinate of the bounding box.
				- max_col (int): Bottom-right column coordinate of the bounding box.
			- ndarray: A smoothed event count map for visualization purposes.
	"""

	height, width = raw.get_size()

	events = raw.load_delta_t(delta_t)
	events['t'] -= int(start_ts)

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
	
	raw.reset()
	return (bounding_boxes, event_count_map_smoothed)
	

def bin_events_over_time(events, total_duration=1e6, small_delta_t=100):
	time_bins = np.arange(0, total_duration + small_delta_t, small_delta_t)
	time_bins_ms = time_bins[:-1] / 1000  #convert to milliseconds

	counts, _ = np.histogram(events['t'], bins=time_bins)

	return counts, time_bins_ms

def detect_peaks_in_event_counts(counts, time_bins_ms):
	peaks, _ = find_peaks(counts, height=np.mean(counts))
	peak_times = time_bins_ms[peaks]

	if len(peak_times) > 1:
		periods = np.diff(peak_times)
		#estimated_frequency = 1000 / np.mean(periods)
	#else:
		#estimated_frequency = np.nan

	return peaks, peak_times

# def perform_fft_analysis(counts, small_delta_t=100):
# 	fft_result = np.fft.fft(counts)
# 	freqs = np.fft.fftfreq(len(counts), d=small_delta_t / 1e6)  #convert delta_t to seconds

# 	idxs = np.where(freqs > 0)
# 	freqs = freqs[idxs]
# 	power = np.abs(fft_result[idxs])

# 	dominant_freq = freqs[np.argmax(power)]

# 	return freqs, power, dominant_freq

def estimate_frequency(events, min_freq = 1000, max_freq = 30000, freq_precision=10, min_count = 0):
	events = np.squeeze(events)
	
	dominant_freq_algo = DominantFrequencyEventsAlgorithm(frequency_precision = freq_precision, min_frequency = min_freq, max_frequency = max_freq, min_count = min_count)
	freq = dominant_freq_algo.compute_dominant_value(events)

	return freq[1]

def estimate_frequency_from_raw(raw, num_events=150000, min_freq = 1000, max_freq = 30000, freq_precision=10, min_count = 0):
	raw.reset()
	evs = raw.load_n_events(num_events)
	
	dominant_freq_algo = DominantFrequencyEventsAlgorithm(frequency_precision = freq_precision, min_frequency = min_freq, max_frequency = max_freq, min_count = min_count)
	freq = dominant_freq_algo.compute_dominant_value(evs)

	return freq[1]

def resample_by_polarity(events):
	'''
		Resamples the signal by sum of event polarities.

		This gives a periodically sampled 1D signal, that can be used for frequency estimation.
	'''

	#raw.reset()

	#events = raw.load_n_events(-1)

	# Create a 1D signal by summing the polarities of events

	# Get the maximum timestamp

	max_t = np.max(events['t'])

	# Create a 1D signal with the same length as the recording

	signal = np.zeros(max_t + 1)

	# Sum the polarities of events at each timestamp

	for event in events:
		signal[event['t']] += event['p']

	return signal

def raws_load_events(raws, dtime = 5000, start_ts = 0.25 * 1e6):
	'''
		Loads events from multiple raw files.

		Args:
			raws: A list of RawReader objects.
			dtime: The time interval (in microseconds) over which to accumulate events for each frame. Default is 500 microseconds.
			start_ts: The starting timestamp (in microseconds) from which to begin processing the event stream. Default is 0.25 seconds.

		Returns:
			A list of event arrays, each containing events from a single raw file.
	'''
	
	events = []

	for raw in raws:
		raw.reset()
		raw.seek_time(int(start_ts))
		evs = raw.load_delta_t(dtime)
		evs['t'] -= int(start_ts)
		events.append(evs)

	return events