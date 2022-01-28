# Purpose of this repo (Abstract)
Deep Neural Networks (DNNs) are the preferred choice for image-based machine learning applications in several domains. However, DNNs are vulnerable to adversarial attacks, that are carefully-crafted perturbations introduced on input data to fool a DNN model. Adversarial attacks may prevent the application of DNNs in security-critical tasks. Consequently, relevant research effort is put in securing DNNs: typical approaches either increase model robustness, or add detection capabilities in the model, or operate on the input data. Instead, in this paper we propose to detect ongoing attacks through monitoring performance indicators of the underlying Graphics Processing Unit (GPU). In fact, adversarial attacks generate images that activate neurons of DNNs in a different way than normal images: we show that this also causes an alteration of GPU activities, that can be observed through software monitors and anomaly detectors. This paper presents our monitoring and detection system, and an extensive experimental analysis that includes a total of 14 adversarial attacks, 3 datasets, and 12 models. We show that, despite limitations on the monitoring granularity, adversarial attacks can be detected in the large majority of cases, with peaks of detection accuracy above 90%.

# Reference

Zoppi, Tommaso, and Andrea Ceccarelli. "Detect Adversarial Attacks Against Deep Neural Networks With GPU Monitoring." IEEE Access 9 (2021): 150579-150591.

# Dataset

https://drive.google.com/drive/folders/15QIz6g2Meun_t_nbwoPf0t5v_PdzEaR5?usp=sharing
(SIZE OF DATASET IS APPROX 140 GB)


# Requirements	
Requires:	
	Linux (tested on Ubuntu 18.04)
	python 3.7 
	NVIDIA GPU
	nvidia-smi tool properly working (usually available; just run nvidia-smi from shell, and see if it works)

# Run the code and reproduce results: instructions
We report how to reproduce results of our work.

Objective is show that we can use GPU parameters to detect adversarial attacks on images.
We execute 14 attacks from the IBM ART Toolbox against mnist, cifar, and stl10 datasets, and on 4 models for each dataset.
The code here allows i) training and creating the datasets with the adversarial images; ii) executing the attacks on the target images; iii) collecting the dataset we use in the paper.

Step i) is very long and generates lots of data --> easiest is to download from google drive.

## Installation	
Download this repo from github (git clone https://github.com/AndreaCeccarelli/gpu-monitor)

Download folders OAIS-images, SAIS-images and savedmodels 
from https://drive.google.com/drive/folders/1a1RCcrJ94oIpqKk_HvRpUAIM3hd13SJu?usp=sharing
and place in the respective directory (you can see them inside the gpu-monitor folder of this github).

Rename folder OAIS-images in "fullattacks"

Rename folder SAIS-images in "synteticattacks"

Install adversarial-robustness-toolbox 1.5.1 from instructions on website.	

Install dependencies through conda package (you can use file gpu-monitor.yml to get an environment called "art2")	
	
Check that GPU is enabled to measure ECC. If not, gpu-monitor.sh may fail. Essentially, what is failing is nvidia-smi, that cannot log the parameters reported in gpu-monitor.sh. In this case, you need to manually remove the problematic parameters from gpu-monitor.sh. Also, remove the same parameters from elaboratedata.py (in the addheader function).
	
## How to run	
If  you just followed the step above, and have the folders fullattacks, savedmodels and synteticattacks with the content from  the google drive, you should only need to set the "--home" flag to make it work. Then you can work on the input parameters, depending on your needs.	

Suggested steps:

conda activate art2	
python pytorch_ART.py --help	
	
Then, use the help configure your settings, e.g., the following should work, just set the right path to the root folder with all the files downloaded from github:

python pytorch_ART.py --attacks_library ./fullattacks/ --home /home/andrea/gpu-monitor/ --full_iterations 1	

(it will take approx. 1-2 hours)	

This will create a set of files under ./dataset . You should look at the *_COMPLETE_DATASET.csv_ files.	

These are the completed files, after all the attacks have been executed.

Now you can test your anomaly detection solution on the CSV file.

You can go in the folder ./evaluations and open target_model.py.
You should change the following paths in target_model.py: 
FILENAME: the name of the "COMPLETE_DATASET.CSV" file you wish to analyze
LOGFILE: where results should be stored
DIR: path to FILENAME.

Then, just run ./target_model.sh . Results will be stored in the LOGFILE.

# Tips (may be dangerous -- be cautious)	
Whenever you experience crashes, we recommend to run a "killall nvidia-smi", to assure that logging is no longer running.	
	
We recommend to configure nvidia-smi with (WARNING: READ MANUAL BEFORE!!! If you are unsure of what you are doing, leave default options! https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf) some flags as follows.

These will require reboot:

- sudo nvidia-smi -e 1/ENABLED

"Set the ECC mode for the target GPUs. See the (GPU ATTRIBUTES) section  for a description of ECC mode. 
Requires root. Will impact all GPUs   unless a single GPU is specified using the -i argument. 
This setting  takes effect after the next reboot and is persistent."
	
- sudo nvidia-smi --gom=0/ALL_ON

"GPU Operation Mode GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.  Each GOM is designed to meet specific user needs.  In ""All On"" everything is enabled and running at full speed. GOM changes take effect after reboot"
	
These are instead disabled after reboot:
- sudo nvidia-smi -pm 1/ENABLED

"Set the persistence mode for the target GPUs. See the (GPU ATTRIBUTES) section for a description of persistence mode. Requires root. Will impact all GPUs unless a single GPU is specified using the -i argument. The effect of this operation is immediate. However, it does not per- sist across reboots. After each reboot persistence mode will default to "Disabled". Available on Linux only."
	
- sudo nvidia-smi -c 0/DEFAULT

"Set the compute mode for the target GPUs. See the (GPU ATTRIBUTES) section for a description of compute mode. Requires root. Will impact all GPUs unless a single GPU is specified using the -i argument. The effect of this operation is immediate. However, it does not persist across reboots. After each reboot compute mode will reset to "DEFAULT""
	
- sudo nvidia-smi -am 1/ENABLED

"Accounting Mode A flag that indicates whether accounting mode is enabled for the GPU Value is either When accounting is enabled statistics are calculated  for each compute process running on the GPU. Statistics can be queried during the lifetime or after termination of the process. The execution  time of process is reported as 0 while the process is in running state and updated to actual execution time after the process has terminated."

# Some Results
Results based on neural networks with entity embedding are in the paper. However, we report results of runs with different algorithms. This shows why we decided to use neural networks amongst the many alternatives available.


![image](https://user-images.githubusercontent.com/20238192/117864377-034a3c00-b295-11eb-8587-bd763fb5faf4.png)
