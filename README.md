# Neural Network Assignment

Dataset taken from [here](https://zindi.africa/competitions/ghana-crop-disease-detection-challenge), you can also find the image dataset here if you want to reproduce the results for yourself


## To Run for yourself: 

 (1) Download the image dataset [here](https://zindi.africa/competitions/ghana-crop-disease-detection-challenge).

 (2) Install requirements:

```pip install -r requirements.txt```

 (3) Run the **Data_Processing.ipynb** notebook (you will need to change the file path to suit your own environment)

 (4) (Optional) Run **compress_images.py** if you are short on memory and processing power. Run it to reduce the size of the images by a lot without much performance and quality loss.

 (5) Run **CNN.ipynb** to run the model that was run in our report, otherwise you can also run of the baseline models **Old_Models.ipynb** (the old resnet model is also at the bottom of Data_Processing.ipynb).


Our random forest baseline is in the Random Forest branch if you want to check that out. Additioanlly ``/vizualizations/`` has some output examples.


**Our Final predictions are stored in Final_Predictions (epoch 20).csv**


 Thank you for using our tool! :heart: 