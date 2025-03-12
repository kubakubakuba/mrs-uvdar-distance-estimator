# import toml

# settings = toml.load('rssr_dataset/settings.toml')

# sequence_length = settings['config']['sequence_length']
# frequency = settings['config']['frequency']
# bias_on = settings['config']['bias_on']
# bias_off = settings['config']['bias_off']
# bias_hpf = settings['config']['bias_hpf']
# camera_height = settings['config']['camera_height']

# print(camera_height)

import os
import toml
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
from collections import defaultdict
from metavision_core.event_io import RawReader
from scipy.signal import find_peaks
from lib import resample_by_polarity

def parse_led_positions(positions_file):
	with open(positions_file, 'r') as f:
		data = toml.load(f)
	
	led_positions = []
	for recording in data['recordings']:
		leds = {}
		for key, value in recording.items():
			if key.startswith('led'):
				led_id = int(key[3:])
				leds[led_id] = np.array([value[0], value[1], value[2]])
		led_positions.append(leds)
	
	filepaths = [recording['filepath'] for recording in data['recordings']]

	return {"led_positions": led_positions, "filepaths": filepaths}

def load_led_events(filepath, led_positions):
	raw = RawReader(filepath)
	raw.seek_time(1)
	evs = raw.load_n_events(-1) #load all
	led_evs = {1: None, 2: None, 3: None, 4: None}

	for i in range(1, 5):
		x, y, roi_size = led_positions.get(i, [None, None, None])
		#filter the roi size
		if x is not None:
			led_evs[i] = evs[(evs['x'] >= x - roi_size) & (evs['x'] <= x + roi_size) & (evs['y'] >= y - roi_size) & (evs['y'] <= y + roi_size)] 
	
	return led_evs

def get_avg_events_per_period(led_evs, freq):
	'''
	Calculates the average number of events per blinking period for a given LED frequency.

	Args:
		led_evs: Structured numpy array of events with 't' (timestamp) and 'p' (polarity) fields.
		freq: Blinking frequency of the LED in Hz.

	Returns:
		Average number of events per period as a float.
	'''
	if led_evs is None:
		return 0.0

	if len(led_evs) == 0:
		return 0.0

	expected_period_us = 1e6 / freq
	
	bins_per_period = 10000
	bin_width_us = expected_period_us / bins_per_period
	bin_width_us = max(bin_width_us, 20)
	bin_width_us = int(bin_width_us)
	
	signal, time_axis = resample_by_polarity(led_evs, bin_width_us)
	
	if len(signal) == 0:
		return 0.0
	
	#print(f"Resampled signal has {len(signal)} samples.")
	distance_samples = expected_period_us / bin_width_us
	distance_samples = max(distance_samples, 1)
	distance_samples = int(distance_samples)
	
	prominence_value = 0.5 * np.max(signal)
	
	peaks, _ = find_peaks(signal, distance=distance_samples, prominence=prominence_value)
	
	if len(peaks) < 2:
		return 0.0
	
	#print(f"Detected {len(peaks)} peaks.")
	peak_times = time_axis[peaks]
	
	event_counts = []
	for i in range(len(peak_times) - 1):
		start_time = peak_times[i]
		end_time = peak_times[i + 1]
		mask = (led_evs['t'] >= start_time) & (led_evs['t'] < end_time)
		event_counts.append(np.sum(mask))
	
	return np.mean(event_counts) if event_counts else 0.0


# Example usage
if __name__ == "__main__":
	freqs = [125, 250, 500, 1000]
	
	dataset_path = "/home/jakub/mrs/git/rssr_dataset"
	
	files = ["positions1.toml", "positions2.toml", "positions3.toml", "positions5.toml"]

	data = []

	for f in files:
		positions_file = os.path.join(dataset_path, f)
		psx1 = parse_led_positions(positions_file)
		
		positions = psx1["led_positions"]
		filepaths = psx1["filepaths"]

		S = []

		for filepath, led_positions in zip(filepaths, positions):
			led_evs = load_led_events(filepath, led_positions)

			avg_1 = get_avg_events_per_period(led_evs[1], freqs[0]/2)
			avg_2 = get_avg_events_per_period(led_evs[2], freqs[1]/2)
			avg_3 = get_avg_events_per_period(led_evs[3], freqs[2]/2)
			avg_4 = get_avg_events_per_period(led_evs[4], freqs[3]/2)

			#avg_1 = len(led_evs[1]) if led_evs[1] is not None else 0
			#avg_2 = len(led_evs[2]) if led_evs[2] is not None else 0
			#avg_3 = len(led_evs[3])	if led_evs[3] is not None else 0
			#avg_4 = len(led_evs[4]) if led_evs[4] is not None else 0

			# print(f"Avg LED1 {avg_1:.2f}")
			# print(f"Avg LED2 {avg_2:.2f}")
			# print(f"Avg LED3 {avg_3:.2f}")
			# print(f"Avg LED4 {avg_4:.2f}")

			stats = [None, None, None, None, None, None]

			try:
				stats[0] = avg_1/avg_2
			except ZeroDivisionError:
				stats[0] = None

			try:
				stats[1] = avg_1/avg_3
			except ZeroDivisionError:
				stats[1] = None

			try:
				stats[2] = avg_1/avg_4
			except ZeroDivisionError:
				stats[2] = None

			try:
				stats[3] = avg_2/avg_3
			except ZeroDivisionError:
				stats[3] = None

			try:
				stats[4] = avg_2/avg_4
			except ZeroDivisionError:
				stats[4] = None

			try:
				stats[5] = avg_4/avg_3
			except ZeroDivisionError:
				stats[5] = None

			#replace inf to 0 and nan to 0
			stats = [0 if x is None else x for x in stats]
			stats = [0 if np.isnan(x) else x for x in stats]
			stats = [0 if np.isinf(x) else x for x in stats]

			
			# print(f"Stats for file {filepath}")

			# print(f"Avg LED2/LED1 {stats[0]}")
			# print(f"Avg LED1/LED3 {stats[1]}")
			# print(f"Avg LED1/LED4 {stats[2]}")
			# print(f"Avg LED2/LED3 {stats[3]}")
			# print(f"Avg LED2/LED4 {stats[4]}")
			# print(f"Avg LED4/LED3 {stats[5]}")

			S.append(stats)

		#average the stats
		S = np.array(S)
		S_avg = np.nanmean(S, axis=0)
		print(f"Average stats")
		print(f"Avg LED2/LED1 {S_avg[0]}")
		print(f"Avg LED1/LED3 {S_avg[1]}")
		print(f"Avg LED1/LED4 {S_avg[2]}")
		print(f"Avg LED2/LED3 {S_avg[3]}")
		print(f"Avg LED2/LED4 {S_avg[4]}")
		print(f"Avg LED4/LED3 {S_avg[5]}")

		#print(get_avg_events_per_period(led_evs[4], 1000))

		data.append(S_avg)

	data = np.array(data)

	plt.figure(figsize=(10, 6))
	plt.plot(data, marker='o')
	plt.title('Average LED Event Ratios')
	plt.xlabel('File Index')
	plt.ylabel('Ratio')
	plt.legend(['LED2/LED1', 'LED1/LED3', 'LED1/LED4', 'LED2/LED3', 'LED2/LED4', 'LED4/LED3'])
	plt.grid(True)
	plt.show()

	distances = range(len(data))
	avg_ratios_per_distance = np.nanmean(data, axis=1)
	plt.figure(figsize=(10, 6))
	plt.plot(distances, avg_ratios_per_distance, marker='o')
	plt.title('Average Ratios at Each Distance')
	plt.xlabel('Distance Index')
	plt.ylabel('Average Ratio')
	plt.grid(True)
	plt.show()