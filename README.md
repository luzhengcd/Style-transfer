# Style Transfer for Pictures and Videos

Style transfer is a technique which allows people to transfer the style of an artistic painting onto any arbitrary picture. It employs convolutional neural network in deep learning to achive the transfer, and could be applied to not only pictures but also videos. This repo is a pytorch implementation of style transfer. 

## Environment configuration

All the codes are written in Python3.6. To quickly install all the packages needed in this project, conda environment management tool is recommended. You can use the environment.yml file we provide to set up the environment if you have conda installed.   

 `conda env create -f environment.yml`
 
 This will create a environment in which all needed packages have been installed. To switch to the environment you just created:   
 `conda activate styleTransfer`  
 `styleTransfer` is the name of the environment, which you can find in `environment.yml`. Click [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details on conda environment management details.
 
