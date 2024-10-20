from metavision_core.event_io import DatWriter, EventsIterator, RawReader
from metavision_sdk_base import EventCD
import numpy as np

def filter_leds(path, areas, output_file):
	"""
		Filter the LEDs in the recordings (cut the areas around the LEDs).

		Args:
			recordings: List of recordings to filter. (each recording is a tuple of (filename, areas))

		Returns:
			None
	"""

	#the areas are now in format (minr, minc, maxr, maxc)

	def filter_events(events, area):
		minc, minr, maxc, maxr = area
		return events[(events['x'] >= minc) & (events['x'] < maxc) & (events['y'] >= minr) & (events['y'] < maxr)]

	def modify_xy(events, area):
		minc, minr, maxc, maxr = area
		events['x'] -= int(minc)
		events['y'] -= int(minr)
		return events

	for idx_a, area in enumerate(areas):
		minr, minc, maxr, maxc = area

		width = maxc - minc
		height = maxr - minr

		# save the filtered events in a file filename_<idx_a>.raw
		f = f"{output_file}_led{str(idx_a)}.dat"

		mv_it = EventsIterator(path, delta_t=10000)

		dat_writer = DatWriter(f, width=int(width), height=int(height))

		for ev in mv_it:
			if ev.size == 0:
				continue

			#filter the events

			ev = filter_events(ev, area)
			ev = modify_xy(ev, area)

			dat_writer.write(ev)

		dat_writer.close()