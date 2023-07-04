# forestClassifier

Multi-label tree species classifier for satellite imagery. See **ForestClassifier.pdf** for detailed description.

TLDR: <br>
**Input** - satellite image of forested area. <br>
**Output** - list of tree species contained in the image.

**Architecture** - CNN and ResNet-50.

**Data** - Sentinel-2 images of Maine, USA; United States Forest Service Forest Inventory and Analysis.

**Results** - A binary accuracy of 83% was achieved.

**Examples:** <br>
The input image can be seen. The true species in the image and the predicted species can be seen underlined in red.

Maine:
<img src="https://github.com/hiddenMedic/forestClassifier/blob/main/figures/predicts/maine/for_T19TCK_img7100_10300.png?raw=true">
<img src="https://github.com/hiddenMedic/forestClassifier/blob/main/figures/predicts/maine/for_T19TCL_img10100_8200.png?raw=true">

Alabama (completely distinct dataset):
<img src="https://github.com/hiddenMedic/forestClassifier/blob/main/figures/predicts/alabama/species_info.png?raw=true">
<img src="https://github.com/hiddenMedic/forestClassifier/blob/main/figures/predicts/alabama/for_img200_4100.png?raw=true">

Serbia (no true data available):
<img src="https://github.com/hiddenMedic/forestClassifier/blob/main/figures/predicts/serbia/for_img10800_9800.png?raw=true">
