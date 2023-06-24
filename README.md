![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/fceba52c-48e2-4d75-82fd-f9ca9b91f527)

# Sales Time Series Forecasting using Machine Learning Techniques (Random Forest, XGBoost, and Stacked Ensemble Regressor)

Developed as a course project in Business Analytics: Operational Research and Risk Analysis program at Alliance Manchester Business School.

# Project Overview

The objective of this project is to build a predictive model to forecast 6 weeks of daily sales for 1,115 drug stores in Europe.

Key steps of the project:
1. Exploratory Data Analysis (EDA)
2. Datetime Objects preprocessing
3. Time Series K-Means clustering using Dynamic Time Warping (to effectively capture curve similarity across time)
4. Generic Preprocessing and Feature Engineering
5. Cluster-specific EDA
6. Variance Inflation Factor (VIF) Backwards Feature Selection (per cluster)
7. Development of Naive Models based on historical sales data (day of week, day of month, day of year)
8. Introduction of historical sales proxy features (Weekly, Monthly based)
9. Three sets of ML models were developed per cluster (No proxy, weekly proxy, monthly proxy)
10. Visualizations of sales predictions for randomly selected stores of each cluster

The ML models used are:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Ensembled Stacking model using Random Forest and XGBoost as weak learners and Linear Regression ad Meta Learner 

The models built for each cluster were evaluated based on Root Mean Square Percentage Error (RMSPE) and R-squared metrics for the training and validation sets. However, RMSPE was primarily used to measure the performance of each model since R-squared might not be suitable for capturing the goodness of the fit of non-linear relationships.

<p align="center">
<img src="https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/8fc90b81-53fe-4d2a-9eff-63ad947f22db">
</p>
<h6>where N is the total number of data records for accuracy measurement, yi is the actual sales for the ith record, ŷi is the sales forecast for the ith record. Zero sales were excluded from the calculation.</h6>

# Installation and Setup

## Codes and Resources Used
- **Editor Used:**  JupyterLab
- **Python Version:** Python 3.9.13

## Python Packages Used
- **General Purpose:** `statsmodels, scipy, time, math`
- **Data Manipulation:** `pandas, numpy` 
- **Data Visualization:** `seaborn, matplotlib`
- **Machine Learning:** `scikit-learn, tslearn`

# Data
- `stores.csv`: contains supplementary information for 1115 stores. (1115 rows, 10 columns)
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/5b518906-b704-40a3-aaf5-5c134215c36e)

- `train.csv`: contains the historical sales data, which covers sales from 01/01/2013 to 31/07/2015. (1017209 rows, 9 columns)
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/d4010000-63d1-4820-9ee4-ad76b7a99340)

- `test.csv`: Identical file to *train.csv*, except that Sales and Customers are unknown for the period of 01/08/2015 to 17/09/2015. (41088 rows, 9 columns)

# Methodology
The overall project design was based on the Cross Industry Standard Process for Data Mining (CRISP-DM).

Figure 1
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/14453db5-5613-42fe-97ab-590191f524ca)

Figure 1 shows the data pre-processing pipeline performed. Data transformation and feature engineering based on the `store_train` data set has been applied across all data sets.

With the pre-processed dataset in hand, clustering was performed to separate the dataset into several clusters for model partitioning. Given the cyclical nature of the sales, Time Series K-Means clustering was employed using Dynamic Time Warping (DTW) as the distance metric to effectively capture curve similarity across time. To remove the effects of differences in the magnitude of sales across stores, the store-specific historical sales data was first transformed by applying *TimeSeriesScalerMeanVariance* to standardize sales on a store-by-store basis, resulting in each store's sales having zero mean and unit variance. 

Additional cluster-specific EDA was then performed on the clustered dataset, aiding in handling missing values and making data feature engineering decisions on each individual cluster. The final clustered training datasets were checked for multicollinearity using the variance inflation factor index (VIF). Features with the highest VIF were iteratively removed until all features had a value less than five. This resulted in the final training, validation, and testing datasets for each cluster.

Figure 2
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/ce99aac2-abc1-4d5a-ac3d-74c541a0997a)

The finalised clustered datasets were then passed for model development, a separate set of models were developed for each cluster as illustrated in Figure 2. Naïve models were first developed as a benchmark for the ML models. The prediction from the naïve models for open stores was either based on the historical weekly average or the monthly average of that specific store, while the prediction for closed stores was 0. Upon retrieving the benchmark results, five selected ML models were developed with and without the `sales_proxy` from the naïve models and further validation was performed. Stacking was then applied on the most robust models to address any potential overfitting issues. To prevent potential data leakage from the `sales_proxy` and one-hot-encoded (day-of-month) variables, validation was exclusively conducted on the validation set, as opposed to adopting a cross-validation approach. Finally, permutation importance method was used to extract the feature importance of the final models and provide business recommendations.

# Results and evaluation
The clustering analysis resulted in 4 clusters of stores(A, B, C, D), and the final features used for each cluster-specific model, after VIF backward selection are shown below. 

![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/e8fcb1dc-393d-4149-badb-00bf04e2d6b0)

## Final Cluster-specific Models 

Initially, a Naive model was developed as a benchmark to test whether the ML methods can provide a more robust forecasting.

Next, 5 base models (MLR, DT, RF, GB, XGB) were developed and it was observed that the non-linear models achieved better results than the Naive ones.

Looking to better improve performance a new feature, sales_proxy (historical sales average for open stores), was introduced within these models. Two different types of models were developed, the Weekly and the Monthly based proxy models, concluding that the Monthly based ones were more powerful, with RF and XGB models outperforming the other ones.

Finally, the Stacking Ensemble method was utilized in order to combine the predictive power of RF and XGB, while also including the Monthly – proxy feature. The results prove that stacking was successful since in most of the clusters the difference between training and validation error decreased, thus reducing overfitting. For the final model selection, RF, XGB, and stacking models were compared. As seen in the table below stacking was chosen for clusters A, C, and D, while RF was chosen for B.

![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/0d0d9fd6-2b7a-41bb-822e-bf35675902c1)

## Permutation Feature Importance for final models across different clusters
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/84690039-9a6b-42fc-a7e0-b4f11188c18c)
![image](https://github.com/MariliaElia/sales-forecast-ml-models/assets/24305018/bb31c5eb-b2fb-40d3-993d-751eb5e25f9d)

The two tables above illustrate the permutation feature importance across the different clusters. First, the model illustrates a strong seasonality pattern. The monthly sales proxy successfully captured a significant amount of the monthly patterns and is the most important feature in terms of permutation importance. The day of the week was also a significant indicator in predicting sales. A trend in the historical data was that sales peaked in December, on Mondays and Sundays, and additionally at the beginning, middle, and end of each month. It was also discovered that school holidays impact sales more than state holidays. Other factors that influenced sales were promotions, with individual store promotions (Promo) appearing to be more effective than coupon-based mailing campaigns (Promo2). The distance to competitors (CompetitionDistance) also showcased some significance within clusters A and C.
