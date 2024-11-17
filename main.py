# Import necessary libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from metavision_core.event_io import RawReader
from scipy.signal import find_peaks

# Set default figure size
plt.rcParams['figure.figsize'] = [8, 6]

# Define utility functions
def recursive_map(func, arr, *args, **kwargs):
	'''
	Recursively applies a function to a nested list and returns the modified list.
	'''
	if isinstance(arr, list):
		return [recursive_map(func, x, *args, **kwargs) for x in arr]
	else:
		return func(arr, *args, **kwargs)

def load_filenames_to_matrix(dataset_path):
	'''
	Loads filenames from the dataset directory and organizes them into a matrix.
	'''
	subdirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
	filenames_matrix = []

	for subdir in subdirs:
		subdir_path = os.path.join(dataset_path, subdir)
		raw_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.raw')])
		filenames_matrix.append([os.path.join(subdir_path, f) for f in raw_files])
	
	return filenames_matrix

def raw_load_events(raw_reader, dtime=10000, start_ts=0.1 * 1e6):
	'''
	Loads events from a raw file using RawReader.
	'''
	raw_reader.seek_time(start_ts)
	events = raw_reader.load_n_events(dtime)
	return events

def resample_by_polarity(events, bin_width_us=1000):
	'''
	Resamples the signal by summing event polarities within fixed time bins.

	Args:
		events: An array of events.
		bin_width_us: Width of each time bin in microseconds.

	Returns:
		signal: A 1D numpy array where each element is the sum of polarities within a bin.
		time_axis: The timestamps corresponding to each bin.
	'''
	if len(events) == 0:
		return np.array([]), np.array([])
	
	timestamps = events['t']
	polarities = events['p']
	
	min_t = np.min(timestamps)
	max_t = np.max(timestamps)
	
	# Define bin edges
	bin_edges = np.arange(min_t, max_t + bin_width_us, bin_width_us)
	
	# Use numpy histogram to bin the data
	signal, _ = np.histogram(timestamps, bins=bin_edges, weights=polarities)
	
	# Create time axis
	time_axis = bin_edges[:-1]  # Left edges of the bins
	
	return signal, time_axis

def detect_blinking_periods(signal, distance_samples, prominence=None):
	'''
	Detect the blinking periods from the resampled signal.

	Args:
		signal: The resampled signal.
		distance_samples: Minimum number of samples between peaks.
		prominence: Required prominence of peaks.

	Returns:
		peak_indices: Indices of the detected peaks in the signal array.
	'''
	peaks, _ = find_peaks(signal, distance=distance_samples, prominence=prominence)
	return peaks

def segment_events_by_peaks(events, time_axis, peak_indices):
	'''
	Segments the events based on the detected peaks.

	Args:
		events: Original events (structured array with 't' field).
		time_axis: Time axis corresponding to the resampled signal.
		peak_indices: Indices of detected peaks in the resampled signal.

	Returns:
		event_counts: List of event counts per detected period.
	'''
	peak_times = time_axis[peak_indices]
	event_counts = []
	
	for i in range(len(peak_times) - 1):
		start_time = peak_times[i]
		end_time = peak_times[i + 1]
		mask = (events['t'] >= start_time) & (events['t'] < end_time)
		count = np.sum(mask)
		event_counts.append(count)
	
	return event_counts

def process_event_data(events, frequency, bin_width_us=1000, prominence_value=None):
	'''
	Processes event data to compute the average number of events per blinking period.

	Args:
		events: Structured array of events with 't' and 'p' fields.
		frequency: Blinking frequency in Hz.
		bin_width_us: Bin width for resampling in microseconds.
		prominence_value: Prominence value for peak detection.

	Returns:
		avg_events: Average number of events per period.
		event_counts: List of event counts per period.
		signal: Resampled signal.
		time_axis: Time axis for the resampled signal.
		peak_indices: Indices of detected peaks.
	'''
	# Resample the events
	signal, time_axis = resample_by_polarity(events, bin_width_us)
	
	# Calculate expected distance between peaks in samples
	expected_period_us = 1e6 / frequency  # Period in microseconds
	distance_samples = expected_period_us / bin_width_us
	
	# Detect peaks
	peak_indices = detect_blinking_periods(signal, distance_samples, prominence=prominence_value)
	
	# Segment events based on peaks
	event_counts = segment_events_by_peaks(events, time_axis, peak_indices)
	
	# Compute average
	avg_events = np.mean(event_counts)
	
	return avg_events, event_counts, signal, time_axis, peak_indices

# Main script
def main():
	# Define dataset path
	dataset_path = "dataset/0/"

	# Load filenames
	filenames = np.array(load_filenames_to_matrix(dataset_path))

	# Load RawReader objects
	raws = []
	for fxs in filenames:
		rawsx = [RawReader(f) for f in fxs]
		raws.append(rawsx)

	# Load events from raw files
	events = recursive_map(raw_load_events, raws, dtime=1000000, start_ts=0.1 * 1e6)  # Adjust dtime and start_ts as needed

	# Define distances and frequencies
	distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 4.0, 5.0]
	frequencies = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000]
	bin_width_us = 1000  # Adjust bin width as needed

	# Process each dataset
	for distance_idx, event_list in enumerate(events):
		print(f"Processing distance {distances[distance_idx]} meters")
		for freq_idx, event_set in enumerate(event_list):
			frequency = frequencies[freq_idx]
			
			if len(event_set) == 0:
				print(f"  No events for frequency {frequency} Hz")
				continue
			
			# Determine prominence value (adjust based on data)
			prominence_value = None  # Start with None and adjust as needed

			# Process event data
			avg_events, event_counts, signal, time_axis, peak_indices = process_event_data(
				event_set,
				frequency,
				bin_width_us=bin_width_us,
				prominence_value=prominence_value
			)
			
			print(f"  Frequency: {frequency} Hz, Average Events per Period: {avg_events}")
			
			# Optionally, visualize the resampled signal and detected peaks
			plt.figure(figsize=(12, 6))
			plt.plot(time_axis, signal)
			plt.plot(time_axis[peak_indices], signal[peak_indices], 'rx')  # Mark peaks
			plt.xlabel('Time (us)')
			plt.ylabel('Summed Polarity')
			plt.title(f'Distance {distances[distance_idx]} m, Frequency {frequency} Hz')
			plt.show()

# Run the main function
if __name__ == "__main__":
	main()
