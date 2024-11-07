from metavision_ml.preprocessing import viz_histo
from metavision_ml.preprocessing.event_to_tensor import histo
from matplotlib import pyplot as plt
from util import bin_events_over_time, detect_peaks_in_event_counts, estimate_frequency, align_signal
import numpy as np
import math

frequency_key_map = {
	10: '__10Hz', 25: '__25Hz', 50: '__50Hz', 100: '__100Hz', 250: '__250Hz',
	500: '__500Hz', 1000: '1k', 2500: '2.5k', 5000: '5k', 10000: '10k', 20000: '20k'
}

def visualize_data_raws(raws, lables, dt = 500, start_ts = 0.5 * 1e6, boxes = None):
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
		ax.set_title(f'Raw data {lables[idx]}', fontsize=20)
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

def visualize_data_events(events_array, lables, dt = 500, start_ts = 0.5 * 1e6, boxes = None):
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
		ax.set_title(f'Event data {lables[idx]}', fontsize=20)
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


def create_combined_plot(events, frequency, total_duration=1e6, small_delta_t=100):
	counts, time_bins_ms = bin_events_over_time(events, total_duration, small_delta_t)
	peaks, peak_times = detect_peaks_in_event_counts(counts, time_bins_ms)
	#freqs, power, dominant_freq = perform_fft_analysis(counts, small_delta_t)

	#frequency = estimate_frequency(events)

	fig, axs = plt.subplots(1, 2, figsize=(25, 5))

	axs[0].plot(time_bins_ms, counts)
	axs[0].set_title('Event Counts Over Time')
	axs[0].set_xlabel('Time (ms)')
	axs[0].set_ylabel('Number of Events')
	axs[0].grid(True)

	axs[1].plot(time_bins_ms, counts)
	axs[1].plot(peak_times, counts[peaks], 'ro')
	axs[1].set_title(f'Event Counts with Detected Peaks\nEstimated Frequency: {frequency:.2f} Hz')
	axs[1].set_xlabel('Time (ms)')
	axs[1].set_ylabel('Number of Events')
	axs[1].grid(True)

	plt.tight_layout()
	plt.show()

def plot_num_events_distance(events, frequencies):
	periods = [math.ceil(1 / (f / 2)) for f in frequencies]

	plt.figure(figsize=(12, 6))

	for idx, frequency in enumerate(frequencies):
		frequency_key = frequency_key_map.get(frequency)
		if frequency_key not in events:
			print(f"Warning: No data for frequency {frequency} Hz")
			continue

		freq_data = events[frequency_key]  # Dictionary with distances as keys
		period = periods[idx]  # Period for current frequency

		sorted_distances = sorted(freq_data.keys(), key=lambda x: float(x.replace("_", ".")))
		datapoints = []
		
		for distance_key in sorted_distances:
			signals = freq_data[distance_key]
			
			inner = []
			for signal in signals:
				aligned_signal = align_signal(signal, period)
				avg_events = np.mean(aligned_signal[:period])
				inner.append(avg_events)

			avg_distance_events = np.average(inner)
			datapoints.append(avg_distance_events)

		plt.plot(datapoints, label=f'Frequency: {frequency} Hz', linestyle='-', marker='x')
	
	plt.xlabel('Index in Resampled Signals (Distance)')
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
	plt.show()