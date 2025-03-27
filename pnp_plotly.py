import numpy as np
import cv2
import toml
from pyocamcalib.modelling.camera import Camera
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative
import os

class PnPlotlySolver:
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

	def load_and_process(self, toml_path: str, rec_num: int = 0):
		"""Load recording data and solve PnP/P3P problem"""
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
		
		res = {
			"rotation_matrix": R,
			"translation_vector": t,
			"distance": np.linalg.norm(t),
			"valid_leds": valid_indices,
			"method": "P3P" if len(valid_indices) == 3 else "PnP",
			"p3p_solutions": p3p_solutions
		}

		return res

	def _get_uav_connections(self, num_points):
		"""Return indices of connected points based on UAV geometry"""
		if num_points == 4:
			return [[0,1], [1,3], [3,2], [2,0], [0,3], [1,2]]  # Full square with diagonals
		elif num_points == 3:
			return [[0,1], [1,2], [2,0]]  # Triangle
		return []

	def _get_rotation_matrix(self, yaw, pitch, roll):
		"""Create rotation matrix from Euler angles (Z-Y-X order)"""
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

	def plot_3d_geometry(self, toml_path: str, rec_num: int = 0, 
						show_all_p3p: bool = False, save_html: bool = False, 
						html_filename: str = "3d_plot.html"):
		"""Interactive 3D visualization using Plotly with ground truth and complete UAV structure."""
		res, recording = self._load_recording_data(toml_path, rec_num)
		image_points, valid_indices = self._get_led_detections(recording)
		gt_points = self._get_ground_truth_points(recording)
		
		fig = go.Figure()
		led_colors = ['red', 'green', 'blue', 'purple']
		
		# 1. Plot camera center
		fig.add_trace(go.Scatter3d(
			x=[0], y=[0], z=[0],
			mode='markers',
			marker=dict(size=10, color='black'),
			name='Camera Center'
		))
		
		# 2. Plot observation rays
		rays = self.cam.cam2world(image_points.copy())
		ray_scale = res["distance"] * 0.75
		for i, (ray, led_idx) in enumerate(zip(rays, valid_indices)):
			scaled_ray = ray / np.linalg.norm(ray) * ray_scale
			fig.add_trace(go.Scatter3d(
				x=[0, scaled_ray[0]], 
				y=[0, scaled_ray[1]], 
				z=[0, scaled_ray[2]],
				mode='lines',
				line=dict(color=led_colors[led_idx], width=4),
				name=f'Ray to LED{led_idx+1}'
			))
		
		# 3. Handle solutions
		if len(valid_indices) == 3 and res["p3p_solutions"] is not None:
			self._plot_3led_case(fig, res, valid_indices, show_all_p3p, led_colors)
		else:
			# 4-LED case
			R = res["rotation_matrix"]
			t = res["translation_vector"]
			projected_points = (R @ self._object_points.T).T + t
			
			# Plot UAV points
			for i, led_idx in enumerate(range(4)):
				fig.add_trace(go.Scatter3d(
					x=[projected_points[i, 0]],
					y=[projected_points[i, 1]],
					z=[projected_points[i, 2]],
					mode='markers',
					marker=dict(size=8, color=led_colors[led_idx]),
					name=f'LED{led_idx+1}'
				))
			
			# Draw UAV structure
			connections = self._get_uav_connections(4)
			for conn in connections:
				fig.add_trace(go.Scatter3d(
					x=projected_points[conn, 0],
					y=projected_points[conn, 1],
					z=projected_points[conn, 2],
					mode='lines',
					line=dict(color='black', width=4),
					showlegend=False
				))
		
		# 4. Plot ground truth if available
		if gt_points is not None:
			# Plot ground truth points
			for i in range(4):
				if i == 0:
					fig.add_trace(go.Scatter3d(
						x=[gt_points[i, 0]],
						y=[gt_points[i, 1]],
						z=[gt_points[i, 2]],
						mode='markers',
						marker=dict(size=8, color='cyan', symbol='diamond'),
						name=f'Ground Truth'
					))
				else:
					fig.add_trace(go.Scatter3d(
						x=[gt_points[i, 0]],
						y=[gt_points[i, 1]],
						z=[gt_points[i, 2]],
						mode='markers',
						marker=dict(size=8, color='cyan'),
						showlegend=False
					))
			
			# Plot ground truth connections
			gt_connections = [[0,1], [1,3], [3,2], [2,0], [0,3], [1,2]]
			for conn in gt_connections:
				fig.add_trace(go.Scatter3d(
					x=gt_points[conn, 0],
					y=gt_points[conn, 1],
					z=gt_points[conn, 2],
					mode='lines',
					line=dict(color='cyan', width=2, dash='dot'),
					showlegend=False
				))
		
		# Set layout
		max_range = res["distance"] * 2
		title = f'3D Geometry - Recording {rec_num}<br>Distance: {res["distance"]:.1f}mm | LEDs: {len(valid_indices)}'
		if gt_points is not None:
			gt_pos = gt_points[0] - self._object_points[0]
			title += f"<br>Ground Truth: X={gt_pos[0]:.0f}mm, Y={gt_pos[1]:.0f}mm, Z={gt_pos[2]:.0f}mm"
		if len(valid_indices) == 3:
			missing_idx = list(set([0,1,2,3]) - set(valid_indices))[0]
			title += f"<br>3-LED case (LED{missing_idx+1} was estimated)"
		
		fig.update_layout(
			title=title,
			scene=dict(
				xaxis=dict(title='X (mm)', range=[-max_range/2, max_range/2]),
				yaxis=dict(title='Y (mm)', range=[-max_range/2, max_range/2]),
				zaxis=dict(title='Z (mm)', range=[0, max_range]),
				aspectmode='manual',
				aspectratio=dict(x=1, y=1, z=0.7),
				camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
			),
			legend=dict(x=1, y=0.5),
			margin=dict(l=0, r=0, b=0, t=40),
			height=800
		)
		
		if save_html:
			fig.write_html(html_filename, include_plotlyjs='../plotly.min.js')
			print(f"Saved interactive plot to {html_filename}")
		
		return fig

	def plot_all_uav_positions(self, toml_path: str, save_html: bool = False, 
							  html_filename: str = "all_uav_positions.html"):
		"""Plot all UAV positions from all recordings in a single 3D plot"""
		with open(toml_path, "r") as f:
			data = toml.load(f)
		
		fig = go.Figure()
		colors = px.colors.qualitative.Plotly * ((len(data["recordings"]) // len(px.colors.qualitative.Plotly)) + 1)
		colors = colors[:len(data["recordings"])]
		
		# Add camera center
		fig.add_trace(go.Scatter3d(
			x=[0], y=[0], z=[0],
			mode='markers',
			marker=dict(size=8, color='black'),
			name='Camera Center'
		))
		
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
				fig.add_trace(go.Scatter3d(
					x=[projected_points[i, 0]], 
					y=[projected_points[i, 1]], 
					z=[projected_points[i, 2]],
					mode='markers',
					marker=dict(size=4, color=colors[rec_num]),
					showlegend=False,
					name=f'Recording {rec_num} LED{led_idx+1}'
				))
			
			# draw UAV connections
			connections = self._get_uav_connections(len(valid_indices))
			for conn in connections:
				fig.add_trace(go.Scatter3d(
					x=projected_points[conn, 0],
					y=projected_points[conn, 1],
					z=projected_points[conn, 2],
					mode='lines',
					line=dict(color=colors[rec_num], width=2),
					showlegend=False,
					name=f'Recording {rec_num} UAV'
				))
			
			centroid = np.mean(projected_points, axis=0)
			all_centroids.append(centroid)
			
			# distance from camera
			distance = np.linalg.norm(centroid)
			all_distances.append(distance)
			
			# Add centroid
			fig.add_trace(go.Scatter3d(
				x=[centroid[0]], 
				y=[centroid[1]], 
				z=[centroid[2]],
				mode='markers',
				marker=dict(size=8, symbol='diamond', color=colors[rec_num]),
				name=f'Recording {rec_num} ({distance:.1f} mm)'
			))
		
		if len(all_distances) > 0:
			avg_distance = np.mean(all_distances)
			std_distance = np.std(all_distances)
			
			print(f"\nDistance Statistics:")
			print(f"Average distance: {avg_distance:.1f} ± {std_distance:.1f} mm")
			print(f"Min distance: {np.min(all_distances):.1f} mm")
			print(f"Max distance: {np.max(all_distances):.1f} mm")
			
			title = (f'All UAV Positions ({len(all_centroids)} recordings)<br>'
					f'Avg distance: {avg_distance:.1f} ± {std_distance:.1f} mm')
		else:
			title = 'No valid recordings found'
		
		if len(all_centroids) > 0:
			# Plot mean position
			mean_position = np.mean(all_centroids, axis=0)
			mean_distance = np.linalg.norm(mean_position)
			fig.add_trace(go.Scatter3d(
				x=[mean_position[0]], 
				y=[mean_position[1]], 
				z=[mean_position[2]],
				mode='markers',
				marker=dict(size=12, symbol='cross', color='red'),
				name=f'Mean Position ({mean_distance:.1f} mm)'
			))
			
			# Set equal aspect ratio based on maximum distance
			max_distance = max(np.linalg.norm(c) for c in all_centroids)
			max_range = max_distance * 2
		
		# Layout configuration
		fig.update_layout(
			title=title,
			scene=dict(
				xaxis=dict(title='X (mm)', range=[-max_range/2, max_range/2] if len(all_centroids) > 0 else None),
				yaxis=dict(title='Y (mm)', range=[-max_range/2, max_range/2] if len(all_centroids) > 0 else None),
				zaxis=dict(title='Z (mm)', range=[0, max_range] if len(all_centroids) > 0 else None),
				aspectmode='manual',
				aspectratio=dict(x=1, y=1, z=0.7),
				camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
			),
			legend=dict(x=1, y=0.5),
			margin=dict(l=0, r=0, b=0, t=40),
			height=800
		)
		
		if save_html:
			fig.write_html(html_filename, include_plotlyjs='../plotly.min.js')
			print(f"Saved interactive plot to {html_filename}")
		
		return fig

	def _load_recording_data(self, toml_path, rec_num):
		"""Load and process recording data."""
		with open(toml_path, "r") as f:
			data = toml.load(f)
		recording = data["recordings"][rec_num]
		res = self.load_and_process(toml_path, rec_num)
		return res, recording

	def _get_led_detections(self, recording):
		"""Extract LED detections from recording."""
		image_points = []
		valid_indices = []
		for i in range(4):
			led_key = f"led{i+1}"
			if led_key in recording and len(recording[led_key]) >= 2:
				image_points.append(recording[led_key][:2])
				valid_indices.append(i)

		return np.array(image_points, dtype=np.float32), valid_indices

	def _get_ground_truth_points(self, recording):
		"""Calculate ground truth points from recording data."""
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
		swivel = recording.get('camera_angle', 0)
		
		# create rotation matrices
		R_pitch_roll = self._get_rotation_matrix(0, pitch, roll)
		R_swivel = np.array([
			[np.cos(np.radians(swivel)), -np.sin(np.radians(swivel)), 0],
			[np.sin(np.radians(swivel)), np.cos(np.radians(swivel)), 0],
			[0, 0, 1]
		])
		R_yaw = np.array([
			[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
			[np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
			[0, 0, 1]
		])
		
		# transform points
		swiveled_position = R_swivel @ gt_position
		uav_center_body = np.mean(self._object_points, axis=0)
		centered_points = self._object_points - uav_center_body
		yawed_points = (R_yaw @ centered_points.T).T + uav_center_body
		gt_points = (R_pitch_roll @ yawed_points.T).T + swiveled_position
		
		return gt_points

	def _plot_3led_case(self, fig, res, valid_indices, show_all_p3p, led_colors):
		"""Handle 3-LED case with Plotly visualization."""
		missing_idx = list(set([0,1,2,3]) - set(valid_indices))[0]
		solution_colors = px.colors.qualitative.Plotly
		
		for sol_num, sol in enumerate(res["p3p_solutions"], 1):
			R = sol["rotation_matrix"]
			t = sol["translation_vector"]
			visible_points = (R @ self._object_points[valid_indices].T).T + t
			
			# Plot visible LEDs
			for i, led_idx in enumerate(valid_indices):
				fig.add_trace(go.Scatter3d(
					x=[visible_points[i, 0]],
					y=[visible_points[i, 1]],
					z=[visible_points[i, 2]],
					mode='markers',
					marker=dict(
						size=8,
						color=led_colors[led_idx],
						opacity=0.7 if show_all_p3p else 1.0
					),
					name=f'LED{led_idx+1}' if sol_num == 1 else None,
					showlegend=sol_num == 1
				))
			
			# Draw triangle
			connections = self._get_uav_connections(3)
			for conn in connections:
				fig.add_trace(go.Scatter3d(
					x=visible_points[conn, 0],
					y=visible_points[conn, 1],
					z=visible_points[conn, 2],
					mode='lines',
					line=dict(
						color=solution_colors[sol_num-1] if show_all_p3p else 'black',
						width=4,
						dash='dot' if show_all_p3p else 'solid'
					),
					showlegend=False
				))
			
			# Estimate and plot missing LED
			missing_pos = self._estimate_missing_led(visible_points, valid_indices, missing_idx)
			fig.add_trace(go.Scatter3d(
				x=[missing_pos[0]],
				y=[missing_pos[1]],
				z=[missing_pos[2]],
				mode='markers',
				marker=dict(
					size=3,
					color=led_colors[missing_idx],
					symbol='x',
					opacity=0.7 if show_all_p3p else 1.0
				),
				name=f'LED{missing_idx+1} (estimated)' if sol_num == 1 else None,
				showlegend=sol_num == 1
			))
			
			# Draw connections to estimated LED
			for i in range(3):
				fig.add_trace(go.Scatter3d(
					x=[missing_pos[0], visible_points[i, 0]],
					y=[missing_pos[1], visible_points[i, 1]],
					z=[missing_pos[2], visible_points[i, 2]],
					mode='lines',
					line=dict(
						color=solution_colors[sol_num-1] if show_all_p3p else 'black',
						width=2,
						dash='dot'
					),
					showlegend=False
				))

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
		
if __name__ == "__main__":
	solver = PnPlotlySolver("calibration.json", uav_size=425)
	dataset_dir = "rssr_dataset"

	plot_dir = "plots1"

	for j in (1, 2, 3, 5):
		#open toml and get the number of recordings
		toml_file = f"positions{j}.toml"
		path = os.path.join(dataset_dir, toml_file)
		
		with open(path, "r") as f:
			data = toml.load(f)

		num_recordings = len(data["recordings"])

		for i in range(num_recordings):
			fig = solver.plot_3d_geometry(path, rec_num=i, show_all_p3p=True, save_html=True, html_filename=f"{plot_dir}/{j}/{i}.html")

	#fig = solver.plot_3d_geometry(path, rec_num=0, show_all_p3p=True, save_html=True, html_filename=f"{plot_dir}/3d_plot.html")
	#fig.show()

	# fig_all = solver.plot_all_uav_positions(path, save_html=True)
	# fig_all.show()