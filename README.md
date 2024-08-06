DROID-SLAM Setup Guide
======================

Overview
--------

DROID-SLAM is designed to work on Linux. Attempts to install it on Windows have been unsuccessful, so it is recommended to use Ubuntu. This guide details the setup process for DROID-SLAM on Ubuntu 22.04 with NVIDIA CUDA 12.2.

Important Notes
---------------

### CUDA Version

Ensure you install the CUDA version that matches your GPU. If your GPU automatically comes with CUDA 12.x, install CUDA 12.x from the official site. Installing a mismatched CUDA version will cause Ubuntu to crash, requiring a complete reinstallation.

### Dual-Boot Requirement

Install Ubuntu alongside Windows in a dual-boot setup. Ubuntu won't function correctly in a virtual machine like VirtualBox for this application. There are many tutorials on YouTube to guide you through the dual-boot setup.

### AMD GPU Compatibility

Please note that if you have an AMD GPU, this setup is unlikely to work.

### Installation Method

Avoid using conda for package installations as it is slow. Use conda only to create the environment and then install all necessary packages using pip for faster performance. Use Python 3.9 when creating your environment. You can also use venv directly from the pip package virtualenv, but note that it may be less stable over time.

Environment Setup Commands
--------------------------

### Virtual Environment Setup

bash

Copy code

`pip install virtualenv
# OR
python -m venv /path/to/new/virtual/environment

virtualenv -p "path/to/python.exe/of/choice" venvname

# Activate the virtual environment
venvname\Scripts\activate
# To deactivate
deactivate`

### Conda Environment Setup

bash

Copy code

`conda create --name droidenv python=3.9

# Activate the conda environment
conda activate droidenv`

### Install Required Packages

bash

Copy code

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ninja
pip install open3d
pip install tensorboard
pip install scipy
pip install opencv-python
pip install tqdm
pip install matplotlib
pip install suitesparse-graphblas
pip install PyYAML
pip install scikit-image
pip install pytransform3d
pip install evo --upgrade --no-binary evo
pip install gdown`

### Additional Setup

If you encounter issues like "python not found", execute the following commands:

bash

Copy code

`sudo apt-get install python3.x-dev # replace .x with your python version, probably 3.9
sudo apt install python3.x-distutils`

Useful Links
------------

-   [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
-   [GCC Setup Issue](https://stackoverflow.com/questions/26053982/setup-script-exited-with-error-command-x86-64-linux-gnu-gcc-failed-with-exit)
-   [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
-   [YouTube Dual-Boot Setup Tutorial](https://www.youtube.com/watch?v=ttxtV966jyQ&t=1695s)
-   [Using Different Python Versions with Virtualenv](https://saturncloud.io/blog/how-to-use-different-python-versions-with-virtualenv/)
-   [Python Installation on Ubuntu](https://www.makeuseof.com/install-python-ubuntu/)
-   [PyCharm Installation on Ubuntu](https://www.javatpoint.com/how-to-install-pycharm-in-ubuntu)
-   [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
-   [PyTorch Geometric Wheels](https://data.pyg.org/whl/)

Dataset Downloads
-----------------

After setting up DROID-SLAM, download some datasets to test its functionality. Here are some links:

-   [Tanks and Temples](https://www.tanksandtemples.org/download/)
-   [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
-   [KITTI Road Dataset](https://www.cvlibs.net/datasets/kitti/eval_road.php)

The Barn link provided in the download_sample_data.sh file does not work; use the Tanks and Temples link instead.

Collecting Your Own Datasets
----------------------------

To collect your own datasets and run inference:

1.  **Required Hardware**: A camera with a 3D sensor (Lidar, stereo camera, etc.). An iPhone with a 3D Lidar sensor can work, but you need the intrinsic parameters of your camera.
2.  **Calibration**: Use the OpenCV chessboard library to find your camera's focal length and center point values for 3D reconstructions. Calibration files are in the form `fx fy cx cy [k1 k2 p1 p2 [ k3 [ k4 k5 k6 ]]]`.
3.  **Professional Cameras**: Using a professional stereo camera like the Intel RealSense simplifies data collection, as these cameras are pre-calibrated and provide depth information, leading to better 3D reconstructions.

Hardware Requirements
---------------------

DROID-SLAM is computationally intensive. You will need a GPU with at least 11GB of VRAM, although it is possible to run on a GPU with 8GB VRAM, such as the NVIDIA Quadro P400. Note that performance may vary, and you could encounter "CUDA ran out of memory" errors. The more VRAM, the better. DROID-SLAM runs smoothly on a 4060TI with 16GB VRAM, while it may lag on a 3080TI with 12GB VRAM.

Additional Resources
--------------------

You can find more detailed discussions and troubleshooting tips on the official DROID-SLAM GitHub issue tracker: [DROID-SLAM Issue #115](https://github.com/princeton-vl/DROID-SLAM/issues/115#issuecomment-1851983842).
