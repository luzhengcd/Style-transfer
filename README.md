# Style Transfer for Pictures and Videos

Style transfer is a technique which allows people to transfer the style of an artistic painting onto any arbitrary picture. It employs convolutional neural network in deep learning to achive the transfer, and could be applied to not only pictures but also videos. This repo is a pytorch implementation of style transfer. 

## Environment configuration

All the codes are written in Python3.6. To quickly install all the packages needed in this project, conda environment management tool is recommended. You can use the environment.yml file we provide to set up the environment if you have conda installed.   

 `conda env create -f environment.yml`
 
 This will create a environment in which all needed packages have been installed. To switch to the environment you just created: `conda activate styleTransfer`. `styleTransfer` is the name of the environment, which you can find in `environment.yml`. Click [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details on conda environment management.

## Implementation details

There are two key parts in this project: transfer net and loss net.

Transfer net is the neural network we need to train. Given a trained model, only a forward pass is needed to stylize any arbitrary picture. We follow []() to build the architecture of the transfer net (see figure 1). Comparing with the optimization-based approach proposed by .., the deep learning approach can achive millisecond style transfer without loss of visual quality of stylized output on a single GPU. 

The loss net is a pre-trained VGG16. VGG19 is also used in some researches. You can use either, but be careful that the content layer and style layer might be different for the two networks. 


## Reference