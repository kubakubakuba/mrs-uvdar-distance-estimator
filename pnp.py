import numpy as np
import cv2
import toml
from pyocamcalib.modelling.camera import Camera
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dill
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.express as px

class PnPsolver:
	def __init__(self, calibration_path: str, uav_size: float = 425):
		self.cam = Camera().load_parameters_json(calibration_path)
		self.uav_size = uav_size
		self._object_points = np.array([
			[uav_size, 0, 0],
			[0, 0, 0],
			[uav_size, uav_size, 0],
			[0, uav_size, 0]
		], dtype=np.float32)
		self.w = 1280
		self.h = 720
		self.p3plabeled = False
	
	def load_and_process(self, toml_path: str, rec_num: int = 0):
		with open(toml_path, "r") as f:
			data = toml.load(f)
		
		recording = data["recordings"][rec_num]
		image_points = []
		valid_indices = []
		
		for i in range(4):
			led_key = f"led{i+1}"
			if led_key in recording and len(recording[led_key]) >= 2:
				image_points.append(recording[led_key][:2])
				valid_indices.append(i)
		
		if len(valid_indices) < 3:
			raise ValueError(f"Need at least 3 points (got {len(valid_indices)})")
		
		image_points = np.array(image_points, dtype=np.float32)
		object_points = self._object_points[valid_indices]
		
		rays = self.cam.cam2world(image_points.copy())
		normalized_points = rays[:, :2] / rays[:, 2].reshape(-1, 1)
		
		print(f"Rays:\n{rays}")
		print(f"Normalized Points:\n{normalized_points}")

		p3p_solutions = []
		
		if len(valid_indices) == 3:
			success, rvecs, tvecs = cv2.solveP3P(
				objectPoints=object_points,
				imagePoints=normalized_points,
				cameraMatrix=np.eye(3),
				distCoeffs=np.zeros((4, 1)),
				flags=cv2.SOLVEPNP_P3P
			)
			if success:
				# Store all solutions
				for i in range(len(rvecs)):
					R, _ = cv2.Rodrigues(rvecs[i])
					t = tvecs[i].flatten()
					p3p_solutions.append({
						"rotation_matrix": R,
						"translation_vector": t,
						"distance": np.linalg.norm(t),
						"solution_num": i+1
					})
				rvec, tvec = rvecs[0], tvecs[0]
			else:
				raise RuntimeError("P3P solution failed")
		else:
			success, rvec, tvec = cv2.solvePnP(
				objectPoints=object_points,
				imagePoints=normalized_points,
				cameraMatrix=np.eye(3),
				distCoeffs=np.zeros((4, 1))
			)
			if not success:
				raise RuntimeError("PnP solution failed")
			p3p_solutions = None
		
		R, _ = cv2.Rodrigues(rvec)
		t = tvec.flatten()
		
		print(f"\nResults for recording {rec_num}:")
		print(f"Valid LEDs: {[i+1 for i in valid_indices]}")
		print(f"Method used: {'P3P' if len(valid_indices) == 3 else 'PnP'}")
		print(f"Rotation Matrix:\n{R}")
		print(f"Translation Vector (mm): {t}")
		print(f"Distance from camera: {np.linalg.norm(t):.1f} mm")

		res = {
			"rotation_matrix": R,
			"translation_vector": t,
			"distance": np.linalg.norm(t),
			"valid_leds": valid_indices,
			"method": "P3P" if len(valid_indices) == 3 else "PnP",
			"p3p_solutions": p3p_solutions  # Include all P3P solutions if available
		}

		return res

	def load_and_process_range(self, toml_path: str, rec_range: range):
		distances = []
		for rec_num in rec_range:
			distances.append(self.load_and_process(toml_path, rec_num)["distance"])

		print(f"\nAverage distance: {np.mean(distances):.1f} mm")
	
	def _get_uav_connections(self, num_points):
		"""Return indices of connected points based on UAV geometry"""
		if num_points == 4:
			return [[0,1], [1,3], [3,2], [2,0], [0,3], [1,2]]  # Full square with diagonals
		elif num_points == 3:
			return [[0,1], [1,2], [2,0]]  # Triangle
		return []
	
	def _get_rotation_matrix(self, yaw, pitch, roll):
		"""Create rotation matrix (Z-Y-X order)"""
		yaw_rad = np.radians(yaw)
		pitch_rad = np.radians(pitch)
		roll_rad = np.radians(roll)
		
		Rz = np.array([
			[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
			[np.sin(yaw_rad), np.cos(yaw_rad), 0],
			[0, 0, 1]
		])
		
		Ry = np.array([
			[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
			[0, 1, 0],
			[-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
		])
		
		Rx = np.array([
			[1, 0, 0],
			[0, np.cos(roll_rad), -np.sin(roll_rad)],
			[0, np.sin(roll_rad), np.cos(roll_rad)]
		])
		
		return Rz @ Ry @ Rx
	
	def plot_all_uav_positions(self, toml_path: str):
		"""Plot all UAV positions from all recordings in a single 3D plot"""
		with open(toml_path, "r") as f:
			data = toml.load(f)
		
		fig = plt.figure(figsize=(12, 10))
		ax = fig.add_subplot(111, projection='3d')
		
		colors = plt.cm.viridis(np.linspace(0, 1, len(data["recordings"])))
		
		ax.scatter([0], [0], [0], c='k', s=100, label='Camera Center')
		
		all_centroids = []
		all_distances = []
		
		for rec_num, recording in enumerate(data["recordings"]):
			valid_indices = []
			for i in range(4):
				led_key = f"led{i+1}"
				if led_key in recording and len(recording[led_key]) >= 2:
					valid_indices.append(i)
			
			if len(valid_indices) < 3:
				continue
			
			res = self.load_and_process(toml_path, rec_num)
			R = res["rotation_matrix"]
			t = res["translation_vector"]
			object_points = self._object_points[valid_indices]
			projected_points = (R @ object_points.T).T + t
			
			# UAV points
			for i, led_idx in enumerate(valid_indices):
				ax.scatter(projected_points[i, 0], projected_points[i, 1], projected_points[i, 2],
						color=colors[rec_num], marker='o', s=30, alpha=0.7)
			
			# draw UAV
			connections = self._get_uav_connections(len(valid_indices))
			for conn in connections:
				ax.plot(projected_points[conn, 0], projected_points[conn, 1], projected_points[conn, 2],
					color=colors[rec_num], alpha=0.3, linewidth=1)
			
			centroid = np.mean(projected_points, axis=0)
			all_centroids.append(centroid)
			
			# distance from camera
			distance = np.linalg.norm(centroid)
			all_distances.append(distance)
			
			# centroid
			ax.scatter(centroid[0], centroid[1], centroid[2],
					color=colors[rec_num], marker='*', s=100,
					label=f'Recording {rec_num} ({distance:.1f} mm)')
		
		if len(all_distances) > 0:
			avg_distance = np.mean(all_distances)
			std_distance = np.std(all_distances)
			
			print(f"\nDistance Statistics:")
			print(f"Average distance: {avg_distance:.1f} ± {std_distance:.1f} mm")
			#print(f"Standard deviation: {std_distance:.1f} mm")
			print(f"Min distance: {np.min(all_distances):.1f} mm")
			print(f"Max distance: {np.max(all_distances):.1f} mm")
			
			title = (f'All UAV Positions ({len(all_centroids)} recordings)\n'
					f'Avg distance: {avg_distance:.1f} ± {std_distance:.1f} mm')
		else:
			title = 'No valid recordings found'
		
		if len(all_centroids) > 0:
			# Plot mean position
			mean_position = np.mean(all_centroids, axis=0)
			mean_distance = np.linalg.norm(mean_position)
			ax.scatter(mean_position[0], mean_position[1], mean_position[2],
					c='r', marker='X', s=200, 
					label=f'Mean Position ({mean_distance:.1f} mm)')
		
		# Set equal aspect ratio based on maximum distance
		if len(all_centroids) > 0:
			max_distance = max(np.linalg.norm(c) for c in all_centroids)
			max_range = max_distance * 1.2
			ax.set_xlim([-max_range/2, max_range/2])
			ax.set_ylim([-max_range/2, max_range/2])
			ax.set_zlim([0, max_range])
		
		# Add coordinate axes
		if len(all_centroids) > 0:
			axis_length = max_range * 0.3
			ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis')
			ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y axis')
			ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z axis')
		
		# Labels and title
		ax.set_xlabel('X (mm)')
		ax.set_ylabel('Y (mm)')
		ax.set_zlabel('Z (mm)')
		ax.set_title(title)
		
		# Set viewing angle and legend
		ax.view_init(elev=25, azim=45)
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax.grid(True)
		
		plt.tight_layout()
		plt.show()

	def plot_3d_geometry(self, toml_path: str, rec_num: int = 0, show_all_p3p: bool = False):
		"""Plot 3D visualization with ground truth and complete UAV structure."""
		res, recording = self._load_recording_data(toml_path, rec_num)
		fig = plt.figure(figsize=(12, 10))
		ax = fig.add_subplot(111, projection='3d')
		
		self._plot_camera_center(ax)
		gt_points = self._plot_ground_truth(ax, recording)
		
		image_points, valid_indices = self._get_led_detections(recording)
		self._plot_observation_rays(ax, image_points, valid_indices)


		if len(valid_indices) == 3 and res["p3p_solutions"] is not None:
			self._process_3led_case(ax, res, image_points, valid_indices, show_all_p3p, gt_points)

		if len(valid_indices) == 4:
			R = res["rotation_matrix"]
			t = res["translation_vector"]
			projected_points = (R @ self._object_points.T).T + t
			
			# plot UAV points
			led_colors = ['red', 'green', 'blue', 'purple']
			for i, led_idx in enumerate(valid_indices):
				ax.scatter(projected_points[i,0], projected_points[i,1], projected_points[i,2],
						color=led_colors[led_idx], marker='o', s=50,
						label=f'LED{led_idx+1}')
			
			# draw full UAV structure
			connections = self._get_uav_connections(4)
			for conn in connections:
				ax.plot(projected_points[conn,0], projected_points[conn,1], projected_points[conn,2],
					'k-', linewidth=2, alpha=0.8)
		
		self._finalize_plot(ax, res, valid_indices, gt_points)
		plt.tight_layout()
		plt.show()

	def _load_recording_data(self, toml_path, rec_num):
		"""Load and process recording data."""
		with open(toml_path, "r") as f:
			data = toml.load(f)
		recording = data["recordings"][rec_num]
		res = self.load_and_process(toml_path, rec_num)
		return res, recording

	def _plot_camera_center(self, ax):
		ax.scatter([0], [0], [0], c='k', s=100, label='Camera Center')

	def _plot_ground_truth(self, ax, recording):
		if not all(k in recording for k in ['dist_x', 'dist_y', 'dist_z']):
			return None

		gt_position = np.array([
			recording['dist_x'],
			recording['dist_y'], 
			recording['dist_z']
		])
		yaw = recording.get('yaw', 0)
		pitch = recording.get('pitch', 0)
		roll = recording.get('roll', 0)
		swivel = recording.get('camera_swivel', 0)  # Get camera swivel angle
		
		# Create rotation matrix for UAV orientation (pitch and roll only)
		R_pitch_roll = self._get_rotation_matrix(0, pitch, roll)
		
		# Apply camera swivel rotation (rotate position around origin)
		R_swivel = np.array([
			[np.cos(np.radians(swivel)), -np.sin(np.radians(swivel)), 0],
			[np.sin(np.radians(swivel)), np.cos(np.radians(swivel)), 0],
			[0, 0, 1]
		])
		swiveled_position = R_swivel @ gt_position
		
		# Calculate UAV center in body frame (center of the 4 points)
		uav_center_body = np.mean(self._object_points, axis=0)
		
		# Apply yaw rotation around UAV center
		R_yaw = np.array([
			[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
			[np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
			[0, 0, 1]
		])
		
		# Transform points:
		# 1. Center points around UAV center
		# 2. Apply yaw rotation
		# 3. Move back to original position
		# 4. Apply pitch/roll rotation
		# 5. Apply position offset
		centered_points = self._object_points - uav_center_body
		yawed_points = (R_yaw @ centered_points.T).T + uav_center_body
		gt_points = (R_pitch_roll @ yawed_points.T).T + swiveled_position
		
		# Plot points (cyan for swiveled ground truth)
		for i in range(4):
			ax.scatter(gt_points[i,0], gt_points[i,1], gt_points[i,2],
					color='cyan', marker='*', s=100,
					label='Ground Truth LEDs' if i == 0 else "")
		
		# Plot connections
		gt_connections = [[0,1], [1,3], [3,2], [2,0], [0,3], [1,2]]
		for conn in gt_connections:
			ax.plot(gt_points[conn,0], gt_points[conn,1], gt_points[conn,2],
				'c-', linewidth=2, alpha=0.5)
		
		# Calculate UAV center in world coordinates
		uav_center_world = np.mean(gt_points, axis=0)
		
		# Add coordinate frame (rotated by yaw, pitch and roll)
		axis_length = 200  # mm
		axes = np.eye(3) * axis_length
		# Apply yaw first to axes, then pitch/roll
		gt_axes = (R_pitch_roll @ (R_yaw @ axes.T)).T + uav_center_world
		
		for i, color in enumerate(['r', 'g', 'b']):
			ax.plot([uav_center_world[0], gt_axes[i,0]],
				[uav_center_world[1], gt_axes[i,1]],
				[uav_center_world[2], gt_axes[i,2]],
				color=color, linewidth=2, alpha=0.7)
		
		return gt_points

	def _get_led_detections(self, recording):
		image_points = []
		valid_indices = []
		for i in range(4):
			led_key = f"led{i+1}"
			if led_key in recording and len(recording[led_key]) >= 2:
				image_points.append(recording[led_key][:2])
				valid_indices.append(i)

		return np.array(image_points, dtype=np.float32), valid_indices

	def _plot_observation_rays(self, ax, image_points, valid_indices):
		ray_scale = 1000  # fixed length for visualization
		led_colors = ['red', 'green', 'blue', 'purple']
		rays = self.cam.cam2world(image_points.copy())
		
		for i, (ray, led_idx) in enumerate(zip(rays, valid_indices)):
			scaled_ray = ray / np.linalg.norm(ray) * ray_scale
			ax.plot([0, scaled_ray[0]], [0, scaled_ray[1]], [0, scaled_ray[2]], 
					color=led_colors[led_idx], alpha=0.7, 
					label=f'Ray to LED{led_idx+1}')

	def _process_3led_case(self, ax, res, image_points, valid_indices, show_all_p3p, gt_points):
		missing_idx = list(set([0,1,2,3]) - set(valid_indices))[0]
		solution_colors = plt.cm.rainbow(np.linspace(0, 1, len(res["p3p_solutions"])))
		led_colors = ['red', 'green', 'blue', 'purple']

		for sol_num, sol in enumerate(res["p3p_solutions"], 1):
			R = sol["rotation_matrix"]
			t = sol["translation_vector"]
			visible_points = (R @ self._object_points[valid_indices].T).T + t
			
			self._plot_solution(ax, sol_num, visible_points, valid_indices, missing_idx, solution_colors, led_colors, show_all_p3p)
			
			missing_pos = self._estimate_missing_led(visible_points, valid_indices, missing_idx)
			self._plot_estimated_led(ax, missing_pos, missing_idx, sol_num, solution_colors, led_colors, show_all_p3p)
			
			self._plot_estimated_connections(ax, missing_pos, visible_points, solution_colors, sol_num, show_all_p3p)
			
			self._calculate_reprojection_errors(image_points, visible_points, missing_pos, missing_idx, sol_num, gt_points, valid_indices)

	def _plot_solution(self, ax, sol_num, visible_points, valid_indices, missing_idx, solution_colors, led_colors, show_all_p3p):
		"""Plot a single P3P solution."""
		alpha = 1.0 if show_all_p3p else 0.3
		color = solution_colors[sol_num-1] if show_all_p3p else 'k'

		for i, led_idx in enumerate(valid_indices):
			if not self.p3plabeled:
				ax.scatter(visible_points[i,0], visible_points[i,1], visible_points[i,2],
						color=led_colors[led_idx], marker='o', s=50, alpha=alpha, label=f'LED{led_idx+1}')
			else:
				ax.scatter(visible_points[i,0], visible_points[i,1], visible_points[i,2],
						color=led_colors[led_idx], marker='o', s=50, alpha=alpha)

		self.p3plabeled = True

		# Draw triangle
		ax.plot(visible_points[[0,1,2,0], 0],
				visible_points[[0,1,2,0], 1],
				visible_points[[0,1,2,0], 2],
				color=color, linewidth=1, alpha=alpha)

	def _estimate_missing_led(self, visible_points, valid_indices, missing_idx):
		"""Estimate position of missing LED using square geometry."""
		idx0, idx1, idx2 = valid_indices
		if missing_idx == 0:  # LED1 missing
			return visible_points[1] - visible_points[2] + visible_points[0]
		elif missing_idx == 1:  # LED2 missing
			return visible_points[0] + visible_points[2] - visible_points[1]
		elif missing_idx == 2:  # LED3 missing
			return visible_points[0] - visible_points[1] + visible_points[2]
		else:  # LED4 missing
			return visible_points[0] + (visible_points[1]-visible_points[0]) + (visible_points[2]-visible_points[0])

	def _plot_estimated_led(self, ax, pos, missing_idx, sol_num, solution_colors, led_colors, show_all_p3p):
		est_color = solution_colors[sol_num-1] if show_all_p3p else led_colors[missing_idx]
		label = f'Sol.{sol_num} LED{missing_idx+1}' if show_all_p3p else f'LED{missing_idx+1} (estimated)'
		ax.scatter(pos[0], pos[1], pos[2],
				color=est_color, marker='x', s=100,
				alpha=0.7, label=label)

	def _plot_estimated_connections(self, ax, missing_pos, visible_points, 
								solution_colors, sol_num, show_all_p3p):
		"""Plot connections between estimated LED and visible LEDs."""
		est_color = solution_colors[sol_num-1] if show_all_p3p else 'k'
		for i in range(3):
			ax.plot([missing_pos[0], visible_points[i,0]],
				[missing_pos[1], visible_points[i,1]],
				[missing_pos[2], visible_points[i,2]],
				'--', color=est_color, linewidth=1, alpha=0.7)

	def _calculate_reprojection_errors(self, image_points, visible_points, missing_pos, missing_idx, sol_num, gt_points, valid_indices):
		all_points = np.vstack([visible_points, missing_pos.reshape(1,3)])
		reprojected = self.cam.world2cam(all_points)

		visible_errors = [np.linalg.norm(reprojected[i] - image_points[i]) for i in range(3)]
		#est_error = np.linalg.norm(reprojected[3] - self.cam.world2cam(self._object_points[missing_idx].reshape(1,3)))

		# calculate the 2d position of the missing led from the 3 image points:
		# TODO: is this OK?
		p1, p2, p3 = image_points
		p4 = p2 + p3 - p1

		est_error = np.linalg.norm(reprojected[3] - p4)
		 
		print(f"\nSolution {sol_num}:")
		print(f"  Visible LED errors: {visible_errors} (avg: {np.mean(visible_errors):.2f} px)")
		print(f"  Estimated LED error: {est_error:.2f} px")

	def _finalize_plot(self, ax, res, valid_indices, gt_points):
		"""Finalize plot with labels, legend, and limits."""
		max_range = res["distance"] * 1.5
		ax.set_xlim([-max_range/2, max_range/2])
		ax.set_ylim([-max_range/2, max_range/2])
		ax.set_zlim([0, max_range])
		
		# Coordinate axes
		axis_length = max_range * 0.3
		ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis')
		ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y axis')
		ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z axis')
		
		ax.set_xlabel('X (mm)')
		ax.set_ylabel('Y (mm)')
		ax.set_zlabel('Z (mm)')
		
		title = f'3D Geometry - Distance: {res["distance"]:.1f} mm'
		if gt_points is not None:
			gt_pos = gt_points[0] - self._object_points[0]  # Get actual position
			title += f"\nGround Truth: X={gt_pos[0]:.0f}mm, Y={gt_pos[1]:.0f}mm, Z={gt_pos[2]:.0f}mm"
		if len(valid_indices) == 3:
			missing_idx = list(set([0,1,2,3]) - set(valid_indices))[0]
			title += f"\n3-LED case (LED{missing_idx+1} estimated)"
		
		ax.set_title(title)
		ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
		ax.grid(True)

	def test_reprojection(self):
		test_point = np.array([[0.001, 0.001, 100]])
		print(self.cam.world2cam(test_point))

if __name__ == "__main__":
	solver = PnPsolver("calibration.json", uav_size=425)
	dataset_dir = "rssr_dataset"
	toml_file = "positions1.toml"

	path = os.path.join(dataset_dir, toml_file)
	with open(path, "r") as f:
		data = toml.load(f)

	num_recordings = len(data["recordings"])

	#solver.load_and_process_range(path, range(num_recordings))

	solver.plot_3d_geometry(path, rec_num=3, show_all_p3p=True)

	#solver.plot_all_uav_positions(path)

	#for r in range(num_recordings):
	#	solver.plot_3d_geometry(path, rec_num=r, show_all_p3p=True)