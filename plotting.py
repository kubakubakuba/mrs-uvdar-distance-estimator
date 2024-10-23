from metavision_ml.preprocessing import viz_histo
from metavision_ml.preprocessing.event_to_tensor import histo
from matplotlib import pyplot as plt
from util import bin_events_over_time, detect_peaks_in_event_counts, estimate_frequency
import numpy as np

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
