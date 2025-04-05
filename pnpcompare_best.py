import numpy as np
import cv2
import toml
import plotly.graph_objects as go
from recipnps.p3p import grunert, fischler, kneip
from recipnps.model import Model
from pyocamcalib.modelling.camera import Camera
import random

class P3PComparatorOptimized:
	def __init__(self, calibration_path: str, uav_size: float = 425):
		self.cam = Camera().load_parameters_json(calibration_path)
		self.uav_size = uav_size
		self._object_points = np.array([
			[uav_size, 0, 0],  # LED1
			[0, 0, 0],          # LED2
			[uav_size, uav_size, 0],  # LED3
			[0, uav_size, 0]    # LED4
		], dtype=np.float32)

	def load_recording(self, toml_path: str, rec_num: int = 0):
		"""Load recording data and extract exactly 3 LED positions"""
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
		
		if len(valid_indices) == 4:
			selected = random.sample(range(4), 3)
			image_points = [image_points[i] for i in selected]
			valid_indices = [valid_indices[i] for i in selected]
		
		return np.array(image_points, dtype=np.float32), valid_indices, recording

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

		swiveled_position = R_swivel @ gt_position
		uav_center_body = np.mean(self._object_points, axis=0)
		centered_points = self._object_points - uav_center_body
		yawed_points = (R_yaw @ centered_points.T).T + uav_center_body
		gt_points = (R_pitch_roll @ yawed_points.T).T + swiveled_position
		
		return gt_points

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

	def compute_reprojection_error(self, R, t, object_points, image_points, fov_degrees=93.5, image_width=1280):
		bearing_vectors = self.cam.cam2world(image_points.copy()).T  # 3xN format
		bearing_vectors = bearing_vectors / np.linalg.norm(bearing_vectors, axis=0)
		
		model = Model(R, t)
		
		cos_errors = model.reprojection_dists_of(object_points.T, bearing_vectors)
		
		angles_rad = np.arccos(1 - cos_errors)
		
		if np.any(angles_rad > np.pi/2):
			return float('inf')
		
		fov_rad = np.radians(fov_degrees)
		pixels_per_radian = image_width / fov_rad
		pixel_errors = angles_rad * pixels_per_radian
		
		return np.mean(pixel_errors)

	def is_real(self, solution):
		return np.allclose(solution.rotation.imag, 0) and np.allclose(solution.translation.imag, 0)

	def solve_opencv_p3p(self, image_points, valid_indices):
		"""Solve using OpenCV's P3P implementation (returns best solution)"""
		object_points = self._object_points[valid_indices]
		
		rays = self.cam.cam2world(image_points.copy())
		normalized_points = rays[:, :2] / rays[:, 2].reshape(-1, 1)
		
		success, rvecs, tvecs = cv2.solveP3P(
			objectPoints=object_points,
			imagePoints=normalized_points,
			cameraMatrix=np.eye(3),
			distCoeffs=np.zeros((4, 1)),
			flags=cv2.SOLVEPNP_P3P
		)
		
		if not success:
			return []
			
		best_solution = None
		min_error = float('inf')
		
		for i in range(len(rvecs)):
			R, _ = cv2.Rodrigues(rvecs[i])
			t = tvecs[i].flatten()

			R = np.real(R)
			t = np.real(t)
			
			error = self.compute_reprojection_error(R, t, object_points, image_points)
			
			print(f"OpenCV P3P solution {i+1}: error = {error:.4f} px, dist = {np.linalg.norm(t):.1f} mm")

			if error < min_error:
				min_error = error
				best_solution = {
					"rotation": R,
					"translation": t,
					"distance": np.linalg.norm(t),
					"method": "OpenCV P3P",
					"reprojection_error": error
				}
		
		return [best_solution] if best_solution is not None else []

	def solve_recipnps_p3p(self, image_points, valid_indices, method: str):
		"""Solve using recipnps P3P methods (returns best solution)"""
		object_points = self._object_points[valid_indices].T  # 3xN format
		
		rays = self.cam.cam2world(image_points.copy())
		image_vectors = rays.T  # 3xN format
		image_vectors = image_vectors / np.linalg.norm(image_vectors, axis=0)

		if method == "grunert":
			solutions = grunert(object_points, image_vectors)
		elif method == "fischler":
			solutions = fischler(object_points, image_vectors)
		elif method == "kneip":
			solutions = kneip(object_points, image_vectors)
		else:
			raise ValueError(f"Unknown method: {method}")
		
		if not solutions:
			return []
			
		best_solution = None
		min_error = float('inf')
		
		for idx, sol in enumerate(solutions):
			if not self.is_real(sol):
				continue

			R = np.real(sol.rotation)
			t = np.real(sol.translation)
			
			error = self.compute_reprojection_error(R, t, object_points.T, image_points)
			
			print(f"{method} solution {idx+1}: error = {error:.4f} px, dist = {np.linalg.norm(t):.1f} mm")
			
			if error < min_error:
				min_error = error
				best_solution = {
					"rotation": R,
					"translation": t,
					"distance": np.linalg.norm(t),
					"method": f"recipnps {method}",
					"reprojection_error": error
				}
		
		return [best_solution] if best_solution is not None else []

	def compare_p3p_methods(self, toml_path: str, rec_num: int = 0):
		"""Compare all P3P methods for a given recording"""
		image_points, valid_indices, recording = self.load_recording(toml_path, rec_num)
		
		print("\n=== Input Points ===")
		print("Image points (pixels):")
		print(image_points)
		print("Object points (mm):")
		print(self._object_points[valid_indices])
		
		results = {
			"opencv": self.solve_opencv_p3p(image_points, valid_indices),
			"grunert": self.solve_recipnps_p3p(image_points, valid_indices, "grunert"),
			"fischler": self.solve_recipnps_p3p(image_points, valid_indices, "fischler"),
			"kneip": self.solve_recipnps_p3p(image_points, valid_indices, "kneip")
		}
		
		return {
			"solutions": results,
			"valid_indices": valid_indices,
			"image_points": image_points,
			"recording": recording
		}
	
	def print_solution_info(self, results):
		"""Print detailed information about all solutions"""
		print("\n=== P3P Solutions Comparison ===")
		
		for method, solutions in results["solutions"].items():
			if not solutions:
				print(f"\n{method.upper():<15} - No solutions found")
				continue
				
			sol = solutions[0]  # the best solution
			print(f"\n{method.upper():<15} - Best Solution:")
			print(f"Method: {sol['method']}")
			print(f"Distance: {sol['distance']:.1f} mm")
			print("Translation (mm):")
			print(f"  X: {sol['translation'][0]:.1f}")
			print(f"  Y: {sol['translation'][1]:.1f}")
			print(f"  Z: {sol['translation'][2]:.1f}")
			
			print("Rotation Matrix:")
			for row in sol["rotation"]:
				print("  [" + " ".join([f"{val:6.3f}" for val in row]) + "]")
			
			sy = np.sqrt(sol["rotation"][0,0]**2 + sol["rotation"][1,0]**2)
			singular = sy < 1e-6
			
			if not singular:
				x = np.arctan2(sol["rotation"][2,1], sol["rotation"][2,2])
				y = np.arctan2(-sol["rotation"][2,0], sy)
				z = np.arctan2(sol["rotation"][1,0], sol["rotation"][0,0])
			else:
				x = np.arctan2(-sol["rotation"][1,2], sol["rotation"][1,1])
				y = np.arctan2(-sol["rotation"][2,0], sy)
				z = 0
			
			print("Euler Angles (degrees):")
			print(f"  Roll (X): {np.degrees(x):.1f}°")
			print(f"  Pitch (Y): {np.degrees(y):.1f}°")
			print(f"  Yaw (Z): {np.degrees(z):.1f}°")
			print(f"Reprojection Error: {sol['reprojection_error']:.4f} px")

	def plot_p3p_comparison(self, toml_path: str, rec_num: int = 0, save_html: bool = False, html_filename: str = "p3p_comparison.html"):
		"""Create 3D plot comparing best P3P solutions with ground truth"""
		results = self.compare_p3p_methods(toml_path, rec_num)
		self.print_solution_info(results)

		valid_indices = results["valid_indices"]
		image_points = results["image_points"]
		recording = results["recording"]
		
		gt_points = self._get_ground_truth_points(recording)
		
		fig = go.Figure()
		
		# camera center
		fig.add_trace(go.Scatter3d(
			x=[0], y=[0], z=[0],
			mode='markers',
			marker=dict(size=10, color='black'),
			name='Camera Center',
			showlegend=True
		))
		
		# plot observation rays
		rays = self.cam.cam2world(image_points.copy())
		ray_length = 4000  # mm
		for i, (ray, led_idx) in enumerate(zip(rays, valid_indices)):
			scaled_ray = ray / np.linalg.norm(ray) * ray_length
			fig.add_trace(go.Scatter3d(
				x=[0, scaled_ray[0]], 
				y=[0, scaled_ray[1]], 
				z=[0, scaled_ray[2]],
				mode='lines',
				line=dict(color='gray', width=2),
				name=f'Ray to LED{led_idx+1}',
				showlegend=True if i == 0 else False
			))
		
		method_colors = {
			"opencv": '#1f77b4',     # blue
			"grunert": '#ff7f0e',    # orange
			"fischler": '#2ca02c',   # green
			"kneip": '#d62728'       # red
		}
		
		for method, solutions in results["solutions"].items():
			if not solutions:
				continue
				
			sol = solutions[0]  # the best solution
			
			# transform object points
			object_points = self._object_points[valid_indices]
			transformed_points = (sol["rotation"] @ object_points.T).T + sol["translation"]
			transformed_points = np.real(transformed_points)
			
			x_vals = np.concatenate([transformed_points[:, 0], [transformed_points[0, 0]]])  # close the triangle
			y_vals = np.concatenate([transformed_points[:, 1], [transformed_points[0, 1]]])
			z_vals = np.concatenate([transformed_points[:, 2], [transformed_points[0, 2]]])
			
			fig.add_trace(go.Scatter3d(
				x=x_vals,
				y=y_vals,
				z=z_vals,
				mode='lines+markers',
				marker=dict(
					size=8,
					color=method_colors[method],
					symbol='circle',
					line=dict(width=1, color='black')
				),
				line=dict(
					color=method_colors[method],
					width=4),
				name=f"{sol['method']} (err={sol['reprojection_error']:.2f} px)",
				showlegend=True
			))
		
		# plot ground truth if available
		if gt_points is not None:
			for i in range(4):
				if i == 0:
					fig.add_trace(go.Scatter3d(
						x=[gt_points[i, 0]],
						y=[gt_points[i, 1]],
						z=[gt_points[i, 2]],
						mode='markers',
						marker=dict(size=8, color='cyan', symbol='diamond'),
						name='Ground Truth'
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
			
			# plot ground truth connections
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
		
		# set layout
		max_range = 2000 * 1.5
		title = f"P3P Methods Comparison - Recording {rec_num}<br>LEDs: {[i+1 for i in valid_indices]}"
		
		if gt_points is not None:
			gt_pos = gt_points[0] - self._object_points[0]
			title += f"<br>Ground Truth: X={gt_pos[0]:.0f}mm, Y={gt_pos[1]:.0f}mm, Z={gt_pos[2]:.0f}mm"
		
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
			legend=dict(
				title='Methods (click to toggle)',
				x=1.05,
				y=0.5,
				itemsizing='constant'
			),
			margin=dict(l=0, r=150, b=0, t=40),
			height=800
		)
		
		if save_html:
			fig.write_html(html_filename, include_plotlyjs='cdn')
			print(f"Saved comparison plot to {html_filename}")
		
		return fig

if __name__ == "__main__":
	comparator = P3PComparatorOptimized("calibration.json", uav_size=425)
	toml_file = "rssr_dataset/positions1.toml"
	
	fig = comparator.plot_p3p_comparison(toml_file, rec_num=2, save_html=True, html_filename="p3p_comparison_optimized.html")
	# fig.show()