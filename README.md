# LightGBM-GBDT-LR
A simple python code of applying GBDT+LR for CTR prediction 

The part of GBDT is proceeded by LightGBM, which is recently proposed by Microsoft, please install it first

https://github.com/Microsoft/LightGBM

The part of Logestic Regression is proceeded by sklearn machine learning.

The main idea is from the work of Facebook published in 2014 that merging GBDT and LR for CTR prediction

http://quinonero.net/Publications/predicting-clicks-facebook.pdf

GBDT is used for feature transformation while the LR uses the transformed data for prediction

