from metavision_ml.preprocessing import histo, viz_histo
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from util import filter_events

def process_raws(raws, labels):
	# Number of histograms
	num_histograms = len(raws)

	# Visualize the areas in the histograms
	rows = 1
	cols = int(np.ceil(num_histograms / rows))
	fig, axes = plt.subplots(rows, cols, figsize=(25, 25))
	axes = axes.flatten()

	for raw in raws:
		raw.reset()

	for idx, raw in enumerate(raws):
		height, width = raw.get_size()
		start_ts = 0.5 * 1e6
		raw.seek_time(start_ts)  # seek in the file to 1s

		delta_t = 500  # sampling duration
		events = raw.load_delta_t(delta_t)  # load 50 milliseconds worth of events
		events['t'] -= int(start_ts)  # important! almost all preprocessing use relative time!

		tbins = 4
		volume = np.zeros((tbins, 2, height, width), dtype=np.float32)
		histo(events, volume, delta_t)

		im = viz_histo(volume[2])
		ax = axes[idx]
		ax.imshow(im)
		ax.set_title(f'Histogram {labels[idx]}', fontsize=20)
		ax.axis('off')

	# Hide any unused subplots
	for ax in axes[num_histograms:]:
		ax.axis('off')

	plt.tight_layout()
	plt.show()


def filter_positive_events(events):
	return events[events['p'] == 0]

def find_freq_and_peaks(raws):
	small_delta_t = 250
	total_duration = 4*1e6

	# Number of bins
	time_bins = np.arange(0, total_duration + small_delta_t, small_delta_t)
	time_bins_ms = time_bins[:-1] / 1000  # convert to milliseconds

	all_event_counts = []

	for idx, raw in enumerate(raws):
		height, width = raw.get_size()
		raw.reset()  # ensure we're at the beginning of the recording
		raw.seek_time(0.1 * 1e6)  # seek to 0.5s

		# Load all events up to total_duration
		events = raw.load_n_events(raw.event_count())  # load all events
		events = events[events['t'] <= total_duration]  # keep events within the total_duration
		events = filter_positive_events(events)  # keep only positive events

		# Bin the events over time
		counts, _ = np.histogram(events['t'], bins=time_bins)
		all_event_counts.append(counts)

		# Create a figure with 3 subplots
		fig, axs = plt.subplots(1, 3, figsize=(16, 5))

		# Plot event counts over time
		axs[0].plot(time_bins_ms, counts)
		axs[0].set_title(f'Event Counts Over Time for Area {idx+1}')
		axs[0].set_xlabel('Time (ms)')
		axs[0].set_ylabel('Number of Events')
		axs[0].grid(True)

		# Peak detection
		peaks, _ = find_peaks(counts, height=np.mean(counts))
		peak_times = time_bins_ms[peaks]
		if len(peak_times) > 1:
			periods = np.diff(peak_times)  # Time between peaks in ms
			estimated_frequency = 1000 / np.mean(periods)  # Frequency in Hz
		else:
			estimated_frequency = np.nan  # Not enough peaks to estimate frequency

		axs[1].plot(time_bins_ms, counts)
		axs[1].plot(peak_times, counts[peaks], 'ro')
		axs[1].set_title(f'Event Counts with Detected Peaks for Area {idx+1}\nEstimated Frequency: {estimated_frequency:.2f} Hz')
		axs[1].set_xlabel('Time (ms)')
		axs[1].set_ylabel('Number of Events')
		axs[1].grid(True)

		# FFT analysis
		fft_result = np.fft.fft(counts)
		freqs = np.fft.fftfreq(len(counts), d=small_delta_t / 1e6)  # Convert delta_t to seconds
		idxs = np.where(freqs > 0)
		freqs = freqs[idxs]
		power = np.abs(fft_result[idxs])

		dominant_freq = freqs[np.argmax(power)]

		axs[2].plot(freqs, power)
		axs[2].set_title(f'FFT of Event Counts for Area {idx+1}\nDominant Frequency: {dominant_freq:.2f} Hz')
		axs[2].set_xlabel('Frequency (Hz)')
		axs[2].set_ylabel('Power')
		axs[2].grid(True)

		plt.tight_layout()
		plt.show()