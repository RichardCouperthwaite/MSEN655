# MSEN655
Code for Design Studio Project

External Link to image data used in the test procedures:
https://drive.google.com/open?id=1YuCaaR5BvnOU0189LqJOMmbHWsbk00Bl

Extract images to directory
./data/crops/

Json file contains index of all images in the directory.

The repository is organised into two sections:
1) Classification: This section deals with the testing of various parameters used in the CNN for classification or regression

  a) For the classification tasks it is necessary to download the image files using the link above and extract them to (./data/crops/) within the classification directory.
  
  b) The controlling function for the code to test the effects of the different structures is the mainCode.py file in the classification directory
  
  c) Depending on GPU capabilities available, it may be necessary to run each structure separately (the code is currently set up to run each model separately, and it will be necessary to change the commented lines to change the models. More information is contained in the mainCode.py file)
    
  
2) Generation: This section deals with testing of parameters for the generation of microstructural images
