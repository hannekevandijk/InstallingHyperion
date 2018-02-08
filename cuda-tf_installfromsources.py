conda create -n tfcuda9.1 python==3.6 numpy pandas matplotlib pydot graphviz scikit-learn

source activate tfcuda9.1

#packages are downloaded to ~/install folder
sudo dpkg -i install/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb 

sudo apt-key add ../../../var/cuda-repo-9-1-local/7fa2af80.pub

sudo apt-get update

sudo apt-get install cuda-9.1

export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo mkdir /usr/lib/systemd/system

#Copy text into file (at nvidia cuda site)
sudo vim /usr/lib/systemd/system/nvidia-persistenced.service
########
#[Unit] 
#Description=NVIDIA Persistence Daemon 
#Wants=syslog.target 

#[Service] 
#Type=forking 
#PIDFile=/var/run/nvidia-persistenced/nvidia-persistenced.pid 
#Restart=always 
#ExecStart=/usr/bin/nvidia-persistenced --verbose 
#ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced 

#[Install] 
#WantedBy=multi-user.target

cd ~
sudo systemctl enable nvidia-persistenced

#comment out memory line that looks like the following
sudo vim /lib/udev/rules.d/40-vm-hotadd.rules
#SUBSYSTEM=="memory", ACTION=="add", PROGRAM="/bin/uname -p", RESULT!="s390*", ATTR{state}=="offline", ATTR{state}="online"

sudo dpkg -i install/libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb 
sudo dpkg -i install/libcudnn7_doc_7.0.5.15-1+cuda9.1_amd64.deb 
sudo dpkg -i install/libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb 

sudo p -r /usr/src/cudnn_samples_v7/mnistCUDNN . 
cd mnistCUDNN
./mnistCUDNN

##this was the cuda installation
##now to building tensorflow

git clone https://github.com/tensorflow/tensorflow

cd tensorflow

source activate tfcuda9.1 #make sure to be in this env

./configure

###
#You have bazel 0.10.0 installed.
#Please specify the location of python. [Default is /home/hanneke/anaconda3/envs/tfcuda9.1/bin/python]: /home/hanneke/anaconda3/envs/tfcuda9.1/bin/python
#
#Found possible Python library paths:
#/home/hanneke/anaconda3/envs/tfcuda9.1/lib/python3.6/site-packages
#Please input the desired Python library path to use. Default is [/home/hanneke/anaconda3/envs/tfcuda9.1/lib/python3.6/site-packages]
#/home/hanneke/anaconda3/envs/tfcuda9.1/lib/python3.6/site-packages
#Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n
#No jemalloc as malloc support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
#No Google Cloud Platform support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
#No Hadoop File System support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
#No Amazon S3 File System support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
#No Apache Kafka Platform support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
#No XLA JIT support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with GDR support? [y/N]: n
#No GDR support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with VERBS support? [y/N]: n
#No VERBS support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
#No OpenCL SYCL support will be enabled for TensorFlow.
#
#Do you wish to build TensorFlow with CUDA support? [y/N]: y
#CUDA support will be enabled for TensorFlow.
#
#Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1
#
#Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda-9.1
#
#Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.0.5
#
#Please specify the location where cuDNN 7.0.5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-9.1]:
#
#Do you wish to build TensorFlow with TensorRT support? [y/N]: n
#No TensorRT support will be enabled for TensorFlow.
#
#Please specify a list of comma-separated Cuda compute capabilities you want to build with.
#You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
#Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 7.0,7.0]7.0,7.0
#
#Do you want to use clang as CUDA compiler? [y/N]: n
#nvcc will be used as CUDA compiler.
#
#Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc
#
#Do you wish to build TensorFlow with MPI support? [y/N]: n
#No MPI support will be enabled for TensorFlow.
#
#Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: n
#
#Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
#Not configuring the WORKSPACE for Android builds.

#Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
#--config=mkl # Build with MKL support.
#--config=monolithic # Config for mostly static monolithic build.
#--config=tensorrt # Build with TensorRT support.
#Configuration finished

bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

sudo /home/hanneke/anaconda3/envs/tfcuda9.1/bin/pip install /tmp/tensorflow_pkg/tensorflow-1.6.0rc0-cp36-cp36m-linux_x86_64.whl

#To check if it works and if it sees the devices, in ipython type:
ipython
##
import tensorflow as tf
tf = tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.list_devices()
exit()
##
