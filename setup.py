"""Setup - single execution before using the tool needed
     * install conda environment
     * modify path variable >> .bashrc
     * chmod spikes_detection
"""
import sys
import os
import logging

if __name__ == '__main__':
    HOME = os.environ['HOME']
    PATH = os.environ['PATH']
    CWD = os.getcwd()
    bashrc_file = os.path.join(HOME, '.bashrc') if os.path.exists(os.path.join(HOME, '.bashrc')) else os.path.join(HOME, '.bash_profile')
    if not os.path.exists(bashrc_file):
        logging.error("environment file does not exist, try manual setup")
        sys.exit()
    else:
        bashrc_file = os.path.basename(bashrc_file)

    if os.path.basename(os.getcwd()) == 'spikes_detection_fcbg':
        os.system("conda env create -f environment.yml")    # installing conda environment
        logging.info("Created conda environment")

        NEW_PATH = f"{CWD}:{PATH}"
        os.system(f"echo 'export PATH={NEW_PATH}' >>  ~/{bashrc_file}")     # add path to the .bashrc file
        os.system(f"source ~/{bashrc_file}")

        os.system(f"chmod u+x spikes_detection")    # give execution rights to the spikes detection code

    else:
        print('Please run setup.py from spikes_detection_fcbg/ directory. Type `cd /path/to/spikes_detection_fcbg/directory` in terminal.')