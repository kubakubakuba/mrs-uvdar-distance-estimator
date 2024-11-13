from metavision_core.event_io import RawReader, EventsIterator
from metavision_sdk_analytics import DominantFrequencyEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_ml.preprocessing import histo, viz_histo

from scipy.signal import find_peaks

import json, os

def load(path):
	#load data json

	json_path = os.path.join(path, 'data.json')
	
	with open(json_path, 'r') as f:
		data = json.load(f)

	directories = data.get('directories', [])
	frequencies = data.get('frequencies', [])
	distances = data.get('distances', [])

	files = []

	for d in directories:
		#for every directory load every .raw file
		raw_files = [os.path.join(path, d, f) for f in os.listdir(os.path.join(path, d)) if f.endswith('.raw')]
		raw_files.sort()
		files.append(raw_files)

	res = {
		'directories': directories,
		'frequencies': frequencies,
		'distances': distances,
		'files': files
	}

	return res