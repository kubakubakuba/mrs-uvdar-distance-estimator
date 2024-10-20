def filter_events(events, area, size):
	x, y = area
	width = size
	height = size
	return events[(events['x'] >= x) & (events['x'] < x + width) & (events['y'] >= y) & (events['y'] < y + height)]