# MSEN655
Code for Design Studio Project

**A Special Note**
The code in this repository relies on randomly selected training and testing data-sets, as such some of the results obtained when running this code may vary from those reported.
**~**

External Link to data used in the test procedures:
https://drive.google.com/open?id=1YuCaaR5BvnOU0189LqJOMmbHWsbk00Bl
External Link for the HDF5 files with the processed image data:
https://drive.google.com/open?id=19BlgjZ2EXhNZ__cUDBWVCLX5fi6edPBy

Extract images from first zip file to directory (Json file contains index of all images in the directory.)
./classification/data/crops/

Extract the HDF5 files in the second zip file to the classification directory


The repository is organised into two sections:
1) Classification: This section deals with the testing of various parameters used in the CNN for classification or regression

  a) For the classification tasks it is necessary to download the image files using the link above and extract them to (./data/crops/) within the classification directory.
  
  b) The controlling function for the code to test the effects of the different structures is the mainCode.py file in the classification directory
  
  c) Ensure that the processedDataHDF5.zip file has been extracted into the classification directory. This zip file contains the processed image data to speed up the importing process and make the code run faster
  
  c) Depending on GPU capabilities available, it may be necessary to run each structure separately (the code is currently set up to run each model separately, and it will be necessary to change the commented lines to change the models. More information is contained in the mainCode.py file.
  
  d) For the optimization of the GP process contained in the final section of the paper, download the finalTest directory from the repository. The code used to generate the various data from the csv files generated by mainCode, is contained within the Data directory, and is title sortlabels.py. All the final csv files are included in the repository, so it shouldn't be necessary to process them again.
  
  e)  The optimization code is written in Matlab. The controlling code is called optimizationCode.m. Running this file will reproduce the results that were obtained in the report, however, some manual post processing will be required to place them in the tables used in the report.
    
  
2) Generation: This section deals with testing of parameters for the generation of microstructural images
