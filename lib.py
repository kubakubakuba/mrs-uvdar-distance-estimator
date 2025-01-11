import os
import numpy as np
from matplotlib import pyplot as plt
from metavision_core.event_io import RawReader
from scipy.signal import find_peaks


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

def process_event_data(events, frequency, prominence_value=None):
	'''
	Processes event data to compute the average number of events per blinking period.
	'''
	# calculate expected period
	expected_period_us = 1e6 / frequency  # period in microseconds
	
	# adjust bin_width_us based on frequency
	bins_per_period = 10  # number of bins per period
	min_bin_width_us = 20  # minimum bin width
	bin_width_us = expected_period_us / bins_per_period
	bin_width_us = max(bin_width_us, min_bin_width_us)
	bin_width_us = int(bin_width_us)
	
	# resample the events
	signal, time_axis = resample_by_polarity(events, bin_width_us)
	
	distance_samples = expected_period_us / bin_width_us
	distance_samples = max(distance_samples, 1)
	distance_samples = int(distance_samples)
	
	# peaks detection
	peak_indices = detect_blinking_periods(signal, distance_samples, prominence=prominence_value)
	
	# segment events based on peaks
	event_counts = segment_events_by_peaks(events, time_axis, peak_indices)
	
	# compute average
	avg_events = np.mean(event_counts)

	# compute mean
	std_events = np.std(event_counts)
	
	res = {
		'avg_events': avg_events,
		'std_events': std_events,
		'event_counts': event_counts,
		'signal': signal,
		'time_axis': time_axis,
		'peak_indices': peak_indices
	}

	return res

def plot_avg_events_vs_distance(distances, avg_events_array, std_events_array, frequencies, title="Influence of Distance on Average Events per Period", save_pgf=False):
	plt.figure()
	
	for freq_idx, frequency in enumerate(frequencies):
		avg_events_per_distance = avg_events_array[:, freq_idx]
		std_events_per_distance = std_events_array[:, freq_idx]
		plt.errorbar(distances, avg_events_per_distance, yerr=std_events_per_distance, marker='o', label=f'{frequency} Hz')
	
	plt.xlabel('Distance (meters)')
	plt.ylabel('Average Events per Blinking Period')
	plt.title(f'{title}')
	plt.legend()
	plt.grid(True)

	if save_pgf:
		print('Saving plot to avg_events_vs_distance.pgf')
		plt.savefig('avg_events_vs_distance.pgf')
	else:
		plt.show()

def plot_log_avg_events_vs_distance(distances, avg_events_array, std_events_array, frequencies, title="Influence of Distance on Log of Average Events per Period", save_pgf=False):
	plt.figure()
	
	for freq_idx, frequency in enumerate(frequencies):
		avg_events_per_distance = avg_events_array[:, freq_idx]
		std_events_per_distance = std_events_array[:, freq_idx]
		plt.errorbar(distances, avg_events_per_distance, yerr=std_events_per_distance, marker='o', label=f'{frequency} Hz')
	
	plt.xlabel('Distance (meters)')
	plt.ylabel('Log of Average Events per Blinking Period')
	plt.title(f'{title}')
	plt.yscale('log')
	plt.legend()
	plt.grid(True)

	if save_pgf:
		print('Saving plot to log_avg_events_vs_distance.pgf')
		plt.savefig('log_avg_events_vs_distance.pgf')
	else:
		plt.show()


def plot_avg_events_vs_frequency(frequencies, avg_events_array, std_events_array, distances, title="Influence of Frequency on Average Events per Period", save_pgf=False):
	plt.figure()
	
	for distance_idx, distance in enumerate(distances):
		avg_events_per_frequency = avg_events_array[distance_idx, :]
		std_events_per_frequency = std_events_array[distance_idx, :]
		plt.errorbar(frequencies, avg_events_per_frequency, yerr=std_events_per_frequency, marker='o', label=f'{distance} m')
	
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Average Events per Blinking Period')
	plt.title(f'{title}')
	plt.xscale('log')
	plt.legend()
	plt.grid(True)
	
	if save_pgf:
		print('Saving plot to avg_events_vs_frequency.pgf')
		plt.savefig('avg_events_vs_frequency.pgf')
	else:
		plt.show()

def plot_log_avg_events_vs_frequency(frequencies, avg_events_array, std_events_array, distances, title="Influence of Frequency on Log of Average Events per Period", save_pgf=False):
	plt.figure()
	
	for distance_idx, distance in enumerate(distances):
		avg_events_per_frequency = avg_events_array[distance_idx, :]
		std_events_per_frequency = std_events_array[distance_idx, :]
		plt.errorbar(frequencies, avg_events_per_frequency, yerr=std_events_per_frequency, marker='o', label=f'{distance} m')
	
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Log of Average Events per Blinking Period')
	plt.title(f'{title}')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.grid(True)

	if save_pgf:
		print('Saving plot to log_avg_events_vs_frequency.pgf')
		plt.savefig('log_avg_events_vs_frequency.pgf')
	else:
		plt.show()

def plot_avg_events_vs_angle(angles, avg_events_array, std_events_array, frequencies, label="", save_pgf=False):
	plt.figure()
	
	for freq_idx, frequency in enumerate(frequencies):
		avg_events_per_angle = avg_events_array[:, freq_idx]
		std_events_per_angle = std_events_array[:, freq_idx]
		plt.errorbar(angles, avg_events_per_angle, yerr=std_events_per_angle, marker='o', label=f'{frequency} Hz')
	
	plt.xlabel('Angle (degrees)')
	plt.ylabel('Average Events per Blinking Period')
	plt.title(f'Influence of Angle on Average Events per Period{label}')
	plt.legend()
	plt.grid(True)
	
	if save_pgf:
		print('Saving plot to avg_events_vs_angle.pgf')
		plt.savefig('avg_events_vs_angle.pgf')
	else:
		plt.show()

def plot_log_avg_events_vs_angle(angles, avg_events_array, std_events_array, frequencies, label="", save_pgf=False):
	plt.figure()
	
	for freq_idx, frequency in enumerate(frequencies):
		avg_events_per_angle = avg_events_array[:, freq_idx]
		std_events_per_angle = std_events_array[:, freq_idx]
		plt.errorbar(angles, avg_events_per_angle, yerr=std_events_per_angle, marker='o', label=f'{frequency} Hz')
	
	plt.xlabel('Angle (degrees)')
	plt.ylabel('Log of Average Events per Blinking Period')
	plt.title(f'Influence of Angle on Log of Average Events per Period{label}')
	plt.yscale('log')
	plt.legend()
	plt.grid(True)
	
	if save_pgf:
		file = f'log_avg_events_vs_angle{label}.pgf'
		file = file.strip()
		file = file.replace(' ', '_')
		file = file.replace('.5', '05')
		file = file.replace('at', '')
		print(f'Saving plot to {file}')
		plt.savefig(file)
	else:
		plt.show()