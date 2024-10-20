python3 -m venv .venv --system-site-packages

export PYTHONNOUSERSITE=true

.venv/bin/python -m pip install pip --upgrade
#.venv/bin/python -m pip install -r /usr/share/metavision/python_requirements/requirements_openeb.txt -r /usr/share/metavision/python_requirements/requirements_sdk_advanced.txt

#.venv/bin/python -m pip install pip --upgrade
.venv/bin/python -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" "h5py==3.7.0" pandas scipy
.venv/bin/python -m pip install matplotlib "ipywidgets==7.6.5"