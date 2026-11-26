nextwave_build.py:
	python make_build_time.py

offline:
	python nextwave_ui.py

all: 
	cmake --build build
