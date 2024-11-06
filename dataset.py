from metavision_core.event_io import RawReader, EventsIterator
from metavision_sdk_analytics import DominantFrequencyEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm
from metavision_ml.preprocessing import histo, viz_histo

import os

from typing import List, Tuple, Dict, Callable

def load_file(f):
	'''
		Loads a raw file from a path.
	
		Args:
			f (str): The path to the raw file.

		Returns:
			A RawReader object.
	'''
	
	if not os.path.exists(f):
		raise FileNotFoundError(f)

	return RawReader(f)

def load_dir(d, subfolders=('0', '45')):
	'''
		Loads all raw files from a directory.
	
		Args:
			d (str): The path to the directory.
			subfolders (list): A list of subfolders to look for raw files.
			
		Returns:
			A dictionary of RawReader objects.
	'''

	raws = {}

	#for each frequency load an array of arrays of raws (for each frequency there is an array of distances, in each of these is an array of raws inside that dir)

	for freq in os.listdir(d):
		freq_path = os.path.join(d, freq)
		if os.path.isdir(freq_path):
			raws[freq] = {}
			#print(f"Processing frequency: {freq}")
			for dist in os.listdir(freq_path):
				dist_path = os.path.join(freq_path, dist)
				if os.path.isdir(dist_path):
					if dist not in raws[freq]:
						raws[freq][dist] = []
					for subfolder in subfolders:
						subfolder_path = os.path.join(dist_path, subfolder)
						if os.path.isdir(subfolder_path):
							raw_files = [f for f in os.listdir(subfolder_path) if f.endswith('.raw')]
							if raw_files:
								#print(f"  Processing subfolder: {subfolder} with files: {raw_files}")
								raws[freq][dist].extend([load_file(os.path.join(subfolder_path, f)) for f in raw_files])
							else:
								print(f"  No .raw files found in subfolder: {subfolder_path}")
						else:
							print(f"  Subfolder path is not a directory: {subfolder_path}")
				else:
					print(f"  Distance path is not a directory: {dist_path}")
		else:
			print(f"Frequency path is not a directory: {freq_path}")

	return raws

def apply(f:Callable, d: Dict, *args, **kwargs):
	'''
		Applies a function to all elements of a dictionary. It also applies the function to all dictionaries in the dictionary.

		Modifies the dictionary in place, does not return anything.
	
		Args:
			d (dict): The dictionary to apply the function to.
			f (function): The function to apply.
			
		Returns:
			None
	'''
	
	for key, value in d.items():
		if isinstance(value, dict):
			apply(f, value, *args, **kwargs)

		elif isinstance(value, list):
			d[key] = [f(v, *args, **kwargs) for v in value]

		else:
			d[key] = f(value, *args, **kwargs)