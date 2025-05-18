#!/usr/bin/env python
import os
import json
import numpy as np
from PnPSolver import PnPSolver
from collections import defaultdict
import csv

class PoseEvaluator:
	def __init__(self, calib_path, data_dir, uav_size=0.425):
		self.solver = PnPSolver(calib_path=calib_path, uav_size=uav_size)
		self.data_dir = data_dir
		self.uav_size = uav_size
		
		self.file_gt_distances = {
			'positions1.json': 1000,
			'positions2.json': 2000,
			'positions3.json': 3000,
			'positions5.json': 5000
		}
		
		self.led_positions = np.array([
			[uav_size, 0, 0],       # LED1
			[0, 0, 0],               # LED2
			[uav_size, uav_size, 0], # LED3
			[0, uav_size, 0]         # LED4
		])
		
	def process_all_recordings(self):
		results = {}
		
		json_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
		
		for json_file in json_files:
			if json_file not in self.file_gt_distances:
				continue
				
			file_path = os.path.join(self.data_dir, json_file)
			file_results = self.process_file(file_path, json_file)
			results[json_file] = file_results
			
		return results
	
	def process_file(self, file_path, filename):
		with open(file_path, 'r') as f:
			data = json.load(f)
			
		file_results = {
			'recordings': [],
			'average_distance_error': 0,
			'average_reprojection_error': 0,
			'num_valid_recordings': 0,
			'gt_distance': self.file_gt_distances[filename] / 1000.0  # Convert to meters
		}
		
		total_distance_error = 0
		total_reprojection_error = 0
		valid_count = 0
		
		for idx, recording in enumerate(data.get('recordings', [])):
			print(f"Processing recording {idx+1}/{len(data['recordings'])} in {filename}")
			result = self.process_recording(recording, file_results['gt_distance'])
			
			if result is not None:
				print(f"Error: {result['distance_error']:.3f} m")
				file_results['recordings'].append(result)
				total_distance_error += result['distance_error']
				total_reprojection_error += result['reprojection_error']
				valid_count += 1
		
		if valid_count > 0:
			file_results['average_distance_error'] = total_distance_error / valid_count
			file_results['average_reprojection_error'] = total_reprojection_error / valid_count
			file_results['num_valid_recordings'] = valid_count
			
		return file_results
	
	def process_recording(self, recording, gt_distance):
		image_points = []
		indices = []
		
		for i in range(4):
			led_key = f"led{i+1}"
			if led_key in recording and len(recording[led_key]) >= 2:
				coords = recording[led_key]
				image_points.append([coords[0], coords[1]])
				indices.append(i)
		
		if len(image_points) < 3:
			return None
			
		solution = self.solver.solve(
			image_points=np.array(image_points),
			indices=np.array(indices),
			verbose=False
		)
		
		if solution is None:
			return None
			
		gt_position = np.array([0, 0, gt_distance])

		est_position = solution['translation']
		
		distance_error = np.linalg.norm(est_position - gt_position)
		position_error = est_position - gt_position
		
		return {
			'filepath': recording.get('filepath', ''),
			'solution': solution,
			'distance_error': distance_error,
			'position_error': position_error.tolist(),
			'reprojection_error': solution['reprojection_error'],
			'used_leds': [i+1 for i in indices],
			'estimated_distance': np.linalg.norm(est_position),
			'gt_distance': gt_distance
		}
	
	def export_for_gnuplot(self, results, output_file="pnp_results.csv"):
		"""Export data in CSV format for gnuplot processing"""
		with open(output_file, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			
			# Write header
			writer.writerow([
				'file', 'recording_id', 'gt_distance', 
				'estimated_distance', 'distance_error',
				'reprojection_error', 'used_leds',
				'x_error', 'y_error', 'z_error'
			])
			
			# Write data rows
			for filename, file_results in results.items():
				for idx, recording in enumerate(file_results['recordings']):
					writer.writerow([
						filename,
						idx+1,
						recording['gt_distance'],
						recording['estimated_distance'],
						recording['distance_error'],
						recording['reprojection_error'],
						'-'.join(map(str, recording['used_leds'])),
						recording['position_error'][0],
						recording['position_error'][1],
						recording['position_error'][2]
					])
		
		print(f"\nData exported to {output_file}")

	def print_results(self, results):
		print("\n=== Evaluation Results ===")
		print(f"Processed {len(results)} files\n")
		
		summary = defaultdict(list)
		
		for filename, file_results in results.items():
			print(f"\nFile: {filename}")
			print(f"Ground Truth Distance: {file_results['gt_distance']:.3f} m")
			print(f"Valid recordings: {file_results['num_valid_recordings']}")
			print(f"Avg distance error: {file_results['average_distance_error']:.3f} m")
			print(f"Avg reprojection error: {file_results['average_reprojection_error']:.2f} px")
			
			if file_results['num_valid_recordings'] > 0:
				summary['distance_errors'].append(file_results['average_distance_error'])
				summary['reprojection_errors'].append(file_results['average_reprojection_error'])
		
		if summary:
			print("\n=== Overall Summary ===")
			print(f"Mean distance error across all files: {np.mean(summary['distance_errors']):.3f} m")
			print(f"Mean reprojection error across all files: {np.mean(summary['reprojection_errors']):.2f} px")
			print(f"Median distance error: {np.median(summary['distance_errors']):.3f} m")
			print(f"Max distance error: {np.max(summary['distance_errors']):.3f} m")
		
		self.export_for_gnuplot(results)

if __name__ == "__main__":
	CALIB_PATH = "calibration.json"
	DATA_DIR = "pnp_plots"
	
	evaluator = PoseEvaluator(calib_path=CALIB_PATH, data_dir=DATA_DIR)
	results = evaluator.process_all_recordings()
	evaluator.print_results(results)