from metavision_ml.preprocessing import viz_histo
from metavision_ml.preprocessing.event_to_tensor import histo
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import math

def visualize_data_raws(raws, labels, dt = 500, start_ts = 0.5 * 1e6, boxes = None):
	num_histograms = len(raws)
	rows = 1
	cols = int(np.ceil(num_histograms / rows))

	fig, axes = plt.subplots(rows, cols, figsize=(25, 25))
	axes = axes.flatten()

	for idx, raw in enumerate(raws):
		raw.reset()
		
		height, width = raw.get_size()
		raw.seek_time(start_ts)

		delta_t = dt  # sampling duration
		events = raw.load_delta_t(delta_t)
		events['t'] -= int(start_ts)

		tbins = 4
		volume = np.zeros((tbins, 2, height, width), dtype=np.float32)
		histo(events, volume, delta_t)

		im = viz_histo(volume[2])
		ax = axes[idx]
		ax.imshow(im)
		ax.set_title(f'Raw data {labels[idx]}', fontsize=20)
		ax.axis('off')

		if boxes != None:
			#boxes come in form of (min_r, min_c, max_r, max_c)
			min_r, min_c, max_r, max_c = boxes[idx]

			x = min_r
			y = min_c
			width = max_c - min_c
			height = max_r - min_r

			ax.add_patch(plt.Rectangle((x, y), width, height, fill=None, edgecolor='red'))

	for ax in axes[num_histograms:]:
		ax.axis('off')

	plt.tight_layout()
	plt.show()

def visualize_data_events(events_array, labels, dt = 500, start_ts = 0.5 * 1e6, boxes = None):
	num_histograms = len(events_array)
	rows = 1
	cols = int(np.ceil(num_histograms / rows))

	fig, axes = plt.subplots(rows, cols, figsize=(25, 25))
	axes = axes.flatten()

	for idx, events_i in enumerate(events_array):
		events = np.array(events_i)[0]

		height = 24
		width = 24
		
		tbins = 4

		volume = np.zeros((tbins, 2, height, width), dtype=np.float32)
		histo(events, volume, dt)


		im = viz_histo(volume[2])
		ax = axes[idx]
		ax.imshow(im)
		ax.set_title(f'Event data {labels[idx]}', fontsize=20)
		ax.axis('off')

		if boxes != None:
			#boxes come in form of (min_r, min_c, max_r, max_c)
			min_r, min_c, max_r, max_c = boxes[idx]

			x = min_r
			y = min_c
			width = max_c - min_c
			height = max_r - min_r

			ax.add_patch(plt.Rectangle((x, y), width, height, fill=None, edgecolor='red'))

	for ax in axes[num_histograms:]:
		ax.axis('off')

	plt.tight_layout()
	plt.show()

def align_signal(signal, period):
	peaks, _ = find_peaks(signal, distance = max(int(period // 2), 1))

	if peaks.size > 0:
		#align to the first detected peak
		start_index = peaks[0]
		return signal[start_index:]
	
	else:
		#if no peaks are found, return the signal as is
		return signal

def plot_avg_events_per_distance(events, resampled, frequencies):
	# Calculate periods for each frequency (in seconds)
	periods = [1 for f in frequencies]  # Periods in seconds

	plt.figure(figsize=(12, 6))

	# Initialize a list of lists to store datapoints for each frequency
	datapoints_by_frequency = [[] for _ in frequencies]

	for idx, distance_events in enumerate(resampled):  # Each distance level
		for freq_idx, signal in enumerate(distance_events):  # Each frequency within that distance

			if len(signal) == 0:
				continue

			event_times = events[idx][freq_idx]['t']
			if len(event_times) == 0:
				continue

			max_t = event_times.max()
			min_t = event_times.min()
			duration = max_t - min_t  # Duration in milliseconds

			if duration <= 0:
				continue

			sampling_rate = len(signal) / duration  # Samples per millisecond
			period_ms = periods[freq_idx] * 1000    # Period in milliseconds
			period_samples = int(period_ms * sampling_rate)

			# Debug statements to verify calculations
			print(f"Distance index: {idx}")
			print(f"Frequency index: {freq_idx}")
			print(f"min_t: {min_t}, max_t: {max_t}")
			print(f"Duration (ms): {duration}")
			print(f"Sampling rate (samples/ms): {sampling_rate}")
			print(f"Period (ms): {period_ms}")
			print(f"Period samples: {period_samples}")

			aligned_signal = align_signal(signal, period_ms)

			if aligned_signal.size >= period_samples:
				avg_events = np.mean(aligned_signal[:period_samples])
			else:
				avg_events = np.mean(aligned_signal) if aligned_signal.size > 0 else 0

			datapoints_by_frequency[freq_idx].append(avg_events)

	# Plot data for each frequency over distances
	for freq_idx, datapoints in enumerate(datapoints_by_frequency):
		if datapoints:  # Only plot if there are datapoints
			plt.plot(datapoints, label=f'Frequency: {frequencies[freq_idx]} Hz', linestyle='-', marker='x')

	plt.xlabel('Distance Index')
	plt.ylabel('Average Number of Events per Period')
	plt.title('Effect of Distance on Number of Events per Blinking Period')
	plt.legend()
	plt.show()

def plot_avg_events_per_frequency(events, frequencies):
	periods = [math.ceil(1 / (f / 2)) for f in frequencies]

	avg_events_per_frequency = []

	for idx, frequency in enumerate(frequencies):
		frequency_key = frequency_key_map.get(frequency)
		if frequency_key not in events:
			print(f"Warning: No data for frequency {frequency} Hz")
			avg_events_per_frequency.append(np.nan)
			continue

		freq_data = events[frequency_key]
		period = periods[idx]

		all_averages = []
		
		for distance_key in freq_data.keys():
			signals = freq_data[distance_key]
			
			inner = []
			for signal in signals:
				aligned_signal = align_signal(signal, period)
				
				if len(aligned_signal) >= period:
					avg_events = np.nanmean(aligned_signal[:period])
					inner.append(avg_events)
				else:
					print(f"Warning: Aligned signal for frequency {frequency} Hz and distance {distance_key} is too short.")
					inner.append(np.nan)
			
			if inner:
				avg_distance_events = np.nanmean(inner)
				all_averages.append(avg_distance_events)

		overall_avg = np.nanmean(all_averages) if all_averages else np.nan
		avg_events_per_frequency.append(overall_avg)

	plt.figure(figsize=(10, 6))
	plt.plot(frequencies, avg_events_per_frequency, marker='o', linestyle='-')
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Average Number of Events per Blinking Period')
	plt.title('Effect of Frequency on Average Number of Events per Blinking Period')
	plt.xscale('log')
	plt.xlim(10, 20000)
	plt.grid(True, which="both", ls="--")
	plt.savefig('avg_events_per_frequency.png')
	plt.show()
