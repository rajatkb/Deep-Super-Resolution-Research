# Deep-Super-Resolution-Research
This repository contains all the work done , observations noted and models worked upon during my internship at [Indian Institute of Technology Gandhinagar](https://www.iitgn.ac.in/) under [Dr. Ravi S. Hegde](https://scholar.google.com/citations?user=aHWM-b8AAAAJ&hl=en). My Research focus was on Image SuperResolution using Deep Learning. Most of my work involved exploring existing solutions and analysing and study their approach. The work was concluded with a Progressive Upsacalling methods using RDNSR with VGG Perceptual Loss for generating high resolution images. The Progressive upscalling can upscale to very high resolution given the RDNSR base network.  

**Report Document For the Inernship submitted in college** : [Report on Deep Super Resolution](https://drive.google.com/open?id=1rY3RL_eTmVhXYWobjMcQ8h8FHuOvjCsj)

* Most of my scripts follow a common pattern and can be understood from these repositories.  
  [RNSR Keras](https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras)  
  [DBPN-SR Keras](https://github.com/rajatkb/Deep-Super-Resolution-Research)  
* The networks given here were worked on 3 Nvidia GTX 1070s with data parallelism in Keras multi gpu mode for training. 
