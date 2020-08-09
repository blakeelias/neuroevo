Running on AWS t2.xlarge (16G RAM). t2.med (4G RAM is too small to render videos!)
Approx 6.25G used. 

## Setup on Master Machine
Only use this venv if you don't want MKL. Otherwise follow the MKL notes conda + venv setup.
```
sudo apt update
sudo apt install virtualenv
virtualenv --python $(which python3) venv
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig  # replaced libavtools with ffpmeg
sudo apt install redis-server
source venv/bin/activate
pip3 install -r requirements.txt
```

To get jupyter notebooks on AWS: https://stackoverflow.com/questions/43241272/can-not-connect-to-jupyter-notebook-on-aws-ec2-instance
```jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
jupyter notebook password
```

### MKL notes
We had to install conda. Install it from here: https://docs.anaconda.com/anaconda/install/linux/
Once it's installed, make sure you open a new shell window.
Then, following instructions from here: https://github.com/IntelPython/mkl-service:
`$conda install -c intel mkl-service`
Test it out:
```
(base) ubuntu@ip-172-31-2-229:~/neuroevo$ python
Python 3.8.3 (default, Jul  2 2020, 16:21:59)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mkl
>>> import tomopy
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'tomopy'
>>> import mkl
>>> mkl.domain_set_num_threads(1, domain='fft') # Intel(R) MKL FFT functions to run sequentially
'success'
>>> import mkl
>>> mkl.set_num_threads(1)
4
>>>
```
Note: `(base)` refers to the base conda environment.

Create a virtualenv that points to the anaconda python. Call this `venvconda3`
```
(base) ubuntu@ip-172-31-2-229:~/neuroevo$ which python3
/home/ubuntu/anaconda3/bin/python3
(base) ubuntu@ip-172-31-2-229:~/neuroevo$ virtualenv --python $(which python3) venvconda3 --system-site-packages
Running virtualenv with interpreter /home/ubuntu/anaconda3/bin/python3
Using base prefix '/home/ubuntu/anaconda3'
/usr/lib/python3/dist-packages/virtualenv.py:1086: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
New python executable in /home/ubuntu/neuroevo/venvconda3/bin/python3
Also creating executable in /home/ubuntu/neuroevo/venvconda3/bin/python
Installing setuptools, pkg_resources, pip, wheel...done.
(base) ubuntu@ip-172-31-2-229:~/neuroevo$ source venvconda3/bin/activate
(venvconda3) (base) ubuntu@ip-172-31-2-229:~/neuroevo$ python
Python 3.8.3 (default, Jul  2 2020, 16:21:59)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mkl
>>> import mkl
>>> mkl.set_num_threads(1)
4
>>>
```
Note: `(venvconda3)` refers to the virtualenv we just created. `(base)` should also be there.


## Create Template 
Create an AWS Machine Image (AMI) based off your master machine. You will use these to spawn workers.

## Launch Worker Instances
  * In AWS console: EC2 > Launch templates
  * For our setup, choose the "Test" template: https://us-east-2.console.aws.amazon.com/ec2/v2/home?region=us-east-2#LaunchTemplateDetails:launchTemplateId=lt-05b0b2a9751f54914
  * Actions > Launch instance from template
  * Set number of instances desired
  * AMI: "neuroevo-worker" (community image)
  * Instance type: c5.18xlarge
  * Keypair: use same keypair as master machine (in our case: "shawnblake")
  * Security groups: use same as master machine (in our case: "launch-wizard-1")
  * Advanced details: 
    * Purchasing option: Request Spot Instances
  
 * Note: If you're using a new AWS EC2 account, you will likely not have the ability to create c5.18xlarge instances. If you attempt to create the instances, you will get the error:
~"The number of VCPUs you're requesting exceeds your limit for this instance type bucket."
You will have to create a support ticket to request an increase in capacity. We requested the ability to create 20x c5.18xlarge spot instances and 20x c5.18xlarge on demand instances.

## Run a Job

Redis server on master must be alive before the redis slave workers. Otherwise, slave workers will exit if unable to attach to the master. Make sure you set the master IP.

* On the master:
`$ nohup redis-server redis.conf`

* On the worker:
  * Edit the settings.py file to have the private IP address from the master. 
    (On the AWS console, go to EC2 > Instances > click on master instance > "Description" tab > Private IPs)

  * `$ do nohup rq worker -c settings > $i` (run a single worker process)

  * Run this command to launch several worker processes:
```
cd ~/neuroevo
source venvconda3/bin/activate
for i in {0..72}
do
nohup rq worker -c settings > $i&
done
```

 * On the master: re-run all cells on the notebook.


