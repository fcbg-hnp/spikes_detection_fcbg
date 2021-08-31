#!/bin/bash

#run make_predictions.py

function make_predictions() {
	cd spikes_detection_fcbg/
 	conda activate spikes_detection
 	python make_predictions.py
	conda deactivate
}
