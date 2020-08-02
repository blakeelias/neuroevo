Running on AWS t2.xlarge (16G RAM). t2.med (4G RAM is too small to render videos!)
Approx 6.25G used. 

sudo apt update
sudo apt install virtualenv
virtualenv --python $(which python3) venv
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig  # replaced libavtools with ffpmeg
sudo apt install redis-server
source venv/bin/activate
pip3 install -r requirements.txt


to get jupyter notebooks on AWS: https://stackoverflow.com/questions/43241272/can-not-connect-to-jupyter-notebook-on-aws-ec2-instance
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py

jupyter notebook password
