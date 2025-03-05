import toml

settings = toml.load('rssr_dataset/settings.toml')

sequence_length = settings['config']['sequence_length']
frequency = settings['config']['frequency']
bias_on = settings['config']['bias_on']
bias_off = settings['config']['bias_off']
bias_hpf = settings['config']['bias_hpf']
camera_height = settings['config']['camera_height']

print(camera_height)