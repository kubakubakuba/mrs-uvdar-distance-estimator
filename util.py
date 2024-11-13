from metavision_core.event_io import RawReader, EventsIterator
from metavision_sdk_analytics import DominantFrequencyEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_ml.preprocessing import histo, viz_histo

import numpy as np

class Signal:
	def __init__(self, signal, min_t, max_t):
		self.signal = signal
		self.min_t = min_t
		self.max_t = max_t
		self.length = len(signal)

	def get_signal(self):
		return self.signal
	
	def get_min_t(self):
		return self.min_t
	
	def get_max_t(self):
		return self.max_t
	
	def get_length(self):
		return self.length

def resample_by_polarity(events):
	'''
		Resamples the signal by sum of event polarities.

		This gives a periodically sampled 1D signal that can be used for frequency estimation.
	
		Args:
			events: An array of events.

		Returns:
			A 1D signal created by summing the polarities of events at each timestamp.
	'''

	#raw.reset()

	#events = raw.load_n_events(-1)

	# Create a 1D signal by summing the polarities of events

	# Get the maximum timestamp

	if len(events) == 0:
		return np.array([])

	max_t = np.max(events['t'])
	min_t = np.min(events['t'])

	# Create a 1D signal with the same length as the recording

	signal = np.zeros(max_t + 1)

	# Sum the polarities of events at each timestamp

	for event in events:
		signal[event['t']] += event['p']

	#res = Signal(signal, min_t, max_t)

	return signal

def raw_load_events(raw, dtime = 5000, start_ts = 0.25 * 1e6):
	'''
		Loads events from a single raw file.

		Args:
			raw: A RawReader object.
			dtime: The time interval (in microseconds) over which to accumulate events for each frame. Default is 500 microseconds.
			start_ts: The starting timestamp (in microseconds) from which to begin processing the event stream. Default is 0.25 seconds.

		Returns:
			An array of events from the raw file.
	'''

	raw.reset()
	raw.seek_time(int(start_ts))
	evs = raw.load_delta_t(dtime)
	evs['t'] -= int(start_ts)

	return evs

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
		events.append(raw_load_events(raw, dtime, start_ts))

	return events
	
def apply(func, arr, *args, **kwargs):
	'''
		x = apply(func, arr, *args, **kwargs)

		Recursively applies a function to a nested list and returns the modified list.
	
		Args:
			func: The function to apply.
			arr: The nested list to modify.
			*args: Additional arguments to pass to the function.
			**kwargs: Additional keyword arguments to pass to the function.
			
		Returns:
			The modified nested list.
	'''
	#recursively apply a function to a nested list and return the modified list

	if isinstance(arr, list):
		return [apply(func, x, *args, **kwargs) for x in arr]
	else:
		return func(arr, *args, **kwargs)