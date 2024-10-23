from metavision_ml.preprocessing import histo, viz_histo
from matplotlib import pyplot as plt
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

	for idx, events in enumerate(events_array):
		events = np.array(events)[0]

		height = 20
		width = 20
		
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