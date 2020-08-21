#!/bin/sh

#  install-packages.sh
#
#
#  Created by Britt D on 8/17/20.
#

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
# set -euo pipefail

# Tell apt-get we're never going to be able to give manual
# feedback:
export DEBIAN_FRONTEND=noninteractive

# Update the package listing, so we know what package exist:
apt-get update

# Install security updates:
apt-get -y upgrade

# Install a new package, without unnecessary recommended packages:
pip3 install syslog-ng
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip3 install pandas==0.25.3
pip3 install numpy==1.17.4
pip3 install seaborn==0.10.0
pip3 install matplotlib==3.1.1
pip3 install scikit-image==0.16.2
pip3 install gym==0.15.4
pip3 install keras==2.2.4
pip3 install tensorflow==1.15.2
pip3 install opencv-python==4.1.1.26
pip3 install scipy==1.3.3
pip3 install joblib==0.14.1
pip3 install coloredlogs==14.0
pip3 install gym[atari]
# need to figure out how to cd to topline baselines and "pip install -e ."

# Delete cached files we don't need anymore:
apt-get clean
rm -rf /var/lib/apt/lists/*
