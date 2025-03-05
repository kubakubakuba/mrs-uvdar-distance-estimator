import argparse
import numpy as np
import cv2
from metavision_core.event_io import RawReader

def accumulate_events(raw_file, start_time_us, accumulation_time_us, output_file, threshold):
	reader = RawReader(raw_file)
	reader.seek_time(start_time_us)

	height, width = reader.get_size()
	pos_accumulator = np.zeros((height, width), dtype=np.uint16)
	neg_accumulator = np.zeros((height, width), dtype=np.uint16)

	end_time_us = start_time_us + accumulation_time_us
	while reader.current_time < end_time_us:
		events = reader.load_delta_t(1000)
		if events is None:
			break

		# Accumulate events
		for event in events:
			x, y, p, _ = event
			if p > 0:
				pos_accumulator[y, x] += 1
			else:
				neg_accumulator[y, x] += 1

	pos_accumulator_normalized = cv2.normalize(pos_accumulator, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	neg_accumulator_normalized = cv2.normalize(neg_accumulator, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	combined_accumulator = cv2.addWeighted(pos_accumulator_normalized, 0.5, neg_accumulator_normalized, 0.5, 0)

	cv2.imwrite(output_file, combined_accumulator)
	print(f"Accumulated events saved to {output_file}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Accumulate events from a raw recording and save as a PNG image.")
	parser.add_argument("--input", type=str, required=True, help="Path to the input raw file.")
	parser.add_argument("--start-time", type=int, required=True, help="Start time in microseconds.")
	parser.add_argument("--accumulation-time", type=int, required=True, help="Accumulation time in microseconds.")
	parser.add_argument("--output", type=str, required=True, help="Output PNG file name.")
	parser.add_argument("--threshold", type=int, default=1, help="Threshold for event accumulation to filter noise.")

	args = parser.parse_args()

	accumulate_events(args.input, args.start_time, args.accumulation_time, args.output, args.threshold)