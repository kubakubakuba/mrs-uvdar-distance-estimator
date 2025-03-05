import cv2
import os
import glob
import typer
import toml
import numpy as np
from pathlib import Path
from generate_frame import accumulate_events
from metavision_core.event_io import RawReader
from metavision_sdk_analytics import DominantFrequencyEventsAlgorithm
from metavision_sdk_core import RoiFilterAlgorithm

app = typer.Typer()

BLOB_RADIUS = 5
BLOB_COLOR = (0, 255, 0)

SELECTED_COLOR = (255, 0, 0)

SELECT_1 = (255, 255, 255)
SELECT_2 = (0, 255, 255)
SELECT_3 = (255, 0, 255)
SELECT_4 = (255, 255, 0)

TEXT_COLOR = (255, 255, 255)

def detect_blob_centers(image_path):
	"""Detect blob centers using contour analysis."""
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		return []
	
	_, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	centers = []
	for contour in contours:
		M = cv2.moments(contour)
		if M["m00"] != 0:
			cx = int(M["m10"] / M["m00"])
			cy = int(M["m01"] / M["m00"])
			centers.append((cx, cy))
	return centers

def estimate_frequency(events, min_freq=1000, max_freq=30000, freq_precision=10, min_count=0):
	"""Estimate the dominant frequency of events."""
	dominant_freq_algo = DominantFrequencyEventsAlgorithm(frequency_precision=freq_precision, min_frequency=min_freq, max_frequency=max_freq, min_count=min_count)
	freq = dominant_freq_algo.compute_dominant_value(events)
	return freq[1]

def process_roi_events(raw_file, x, y, roi_size):
	"""Process events within a region of interest (ROI)."""
	roi_filter = RoiFilterAlgorithm(
		x - roi_size, y - roi_size, x + roi_size, y + roi_size
	)

	reader = RawReader(raw_file)

	filtered_events = []
	while not reader.done:
		events = reader.load_delta_t(10000)
		if events.size == 0:
			continue

		output_buf = roi_filter.get_empty_output_buffer()
		roi_filter.process_events(events, output_buf)

		if output_buf.numpy().size > 0:
			filtered_events.append(output_buf.numpy())

	if filtered_events:
		return np.concatenate(filtered_events)
	else:
		return np.array([], dtype=np.dtype([('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')]))
	
@app.command()
def main(
	input_folder: str = typer.Argument(..., help="Folder containing raw recordings"),
	output_file: str = typer.Argument(..., help="Output TOML file"),
	accumulation_time_us: int = typer.Option(1000000, help="Global accumulation time in microseconds"),
	threshold: int = typer.Option(1, help="Threshold for event accumulation"),
	roi_size: int = typer.Option(0, help="Size of the region of interest (ROI) in pixels"),
	detect_blobs: bool = typer.Option(
		True,
		"--no-detect",
		help="Detect blobs in images.",
	),
):
	"""
	Process raw event recordings and label LED positions.
	
	Key controls:
	u/i/o/p - Select LED 1/2/3/4 before clicking
	n - Create new blob at mouse position
	r - Reset selections
	c - Confirm and continue
	F - Estimate frequency for selected points
	ESC - Exit program
	"""
	
	recordings = []
	raw_files = sorted(glob.glob(os.path.join(input_folder, "*.raw")))

	for idx, raw_file in enumerate(raw_files):
		print(f"Opening recording {raw_file}, {idx + 1} of {len(raw_files)}")
		temp_image_path = "temp_frame.png"
		accumulate_events(raw_file, 0, accumulation_time_us, temp_image_path, threshold)
		
		blob_centers = []
		if detect_blobs:
			blob_centers = detect_blob_centers(temp_image_path)
			original_centers = blob_centers.copy()
		
		image = cv2.imread(temp_image_path)
		if image is None:
			continue
			
		image_display = image.copy()
		for center in blob_centers:
			cv2.circle(image_display, center, BLOB_RADIUS, BLOB_COLOR, -1)
		
		selected_leds = {}
		current_led = None
		mouse_pos = (0, 0)

		def mouse_callback(event, x, y, flags, param):
			nonlocal mouse_pos, blob_centers, selected_leds, current_led, image_display
			mouse_pos = (x, y)
			if event == cv2.EVENT_LBUTTONDOWN and current_led is not None:
				min_dist = float('inf')
				closest = None
				for center in blob_centers:
					dist = (center[0] - x)**2 + (center[1] - y)**2
					if dist < min_dist:
						min_dist = dist
						closest = center
				if closest is not None:
					selected_leds[current_led] = closest
					if current_led == 1:
						color = SELECT_1
					elif current_led == 2:
						color = SELECT_2
					elif current_led == 3:
						color = SELECT_3
					elif current_led == 4:
						color = SELECT_4
					else:
						color = SELECTED_COLOR
					cv2.circle(image_display, closest, BLOB_RADIUS, color, -1)
					cv2.putText(image_display, f"LED{current_led}", 
							   (closest[0]+10, closest[1]+10), 
							   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
					current_led = None
					cv2.imshow("LED Labeling", image_display)

					if roi_size > 0:
						cv2.rectangle(image_display, (x-roi_size, y-roi_size), (x+roi_size, y+roi_size), (255, 255, 255), 1)
						cv2.imshow("LED Labeling", image_display)

		cv2.namedWindow("LED Labeling")
		cv2.setMouseCallback("LED Labeling", mouse_callback)
		cv2.imshow("LED Labeling", image_display)

		while True:
			key = cv2.waitKey(1) & 0xFF

			if key == ord('u'):
				current_led = 1
				print(f"Selecting LED {current_led}")
			
			elif key == ord('i'):
				current_led = 2
				print(f"Selecting LED {current_led}")
			
			elif key == ord('o'):
				current_led = 3
				print(f"Selecting LED {current_led}")
			
			elif key == ord('p'):
				current_led = 4
				print(f"Selecting LED {current_led}")

			elif key == ord('n'):
				blob_centers.append(mouse_pos)
				cv2.circle(image_display, mouse_pos, BLOB_RADIUS, BLOB_COLOR, -1)
				cv2.imshow("LED Labeling", image_display)
			elif key == ord('r'):
				blob_centers = original_centers.copy()
				selected_leds = {}
				current_led = None
				image_display = image.copy()
				for center in blob_centers:
					cv2.circle(image_display, center, BLOB_RADIUS, BLOB_COLOR, -1)
				cv2.imshow("LED Labeling", image_display)
			elif key == ord('c'):
				break
			elif key == ord('f'):
				for led, (x, y) in selected_leds.items():
					filtered_events = process_roi_events(raw_file, x, y, roi_size)
					if filtered_events.size > 0:
						freq = estimate_frequency(filtered_events)
						print(f"LED {led} frequency: {freq} Hz")
					else:
						print(f"No events found in ROI for LED {led}.")
			elif key == 27:  # ESC key
				cv2.destroyAllWindows()
				return

		cv2.destroyAllWindows()

		recording_entry = {"filepath": str(Path(raw_file).resolve())}
		for led in [1, 2, 3, 4]:
			if led in selected_leds:
				x, y = selected_leds[led]
				recording_entry[f"led{led}"] = [int(x), int(y), int(roi_size)]
			else:
				recording_entry[f"led{led}"] = None
		
		recordings.append(recording_entry)
		os.remove(temp_image_path)

	with open(output_file, "w") as f:
		toml.dump({"recordings": recordings}, f)
	typer.echo(f"Saved labeling data to {output_file}")

if __name__ == "__main__":
	app()