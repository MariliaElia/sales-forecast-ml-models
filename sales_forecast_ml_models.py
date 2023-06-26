## Sales Time Series Forecasting using Machine Learning Techniques (Random Forest, XGBoost, Stacked Ensemble Regressor)

# ## Part 0 - Preliminary
# ### Part 0.1 - Library Imports

#basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

#statistics libraries
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

#preprocessing library
from sklearn.preprocessing import OneHotEncoder

#clustering libraries
from sklearn.cluster import KMeans
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

#model libraries
#elementary
from sklearn.linear_model import LinearRegression

#tree-based
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

#other models
from sklearn.ensemble import StackingRegressor

#r2_score
from sklearn.metrics import r2_score

#Setting the random seed
import random
seed = 42
np.random.seed(seed)
random.seed(seed)

import warnings
warnings.filterwarnings('ignore')


# ### Part 0.2 - Data Importing
train = pd.read_csv("./data/DA2023_train.csv")
train["Date"] = pd.to_datetime(train["Date"], infer_datetime_format = True)
train.head()

test = pd.read_csv("./data/DA2023_test.csv")
test["Date"] = pd.to_datetime(test["Date"], infer_datetime_format = True)
test.head()

stores = pd.read_csv("./data/DA2023_stores.csv")
stores.head()

stores = stores[stores.columns[:-2]]
print(stores.shape)
stores.columns


# ## Part 1 - Clustering and Model Partition
# Initial Merging for EDA
stores_train = train.merge(stores, how = "left", on = "Store")
stores_train.head()


# ### Part 1.1 EDA for Naive Clustering
def boxplot(x,y,data,title,xlabel,ylabel): #function for grouped boxplots

    box_plot = sns.boxplot(x=x, y=y, data=data)

    #modify individual font size of elements
    sns.set(rc = {'figure.figsize':(20, 7)})
    plt.xlabel(xlabel, fontsize=20);
    plt.ylabel(ylabel, fontsize=20,);
    plt.title(title, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=17)

boxplot('Assortment','Sales',stores_train,'Distribution of Sales by Assortment','Assortment','Sales')

boxplot('StoreType','Sales',stores_train,'Distribution of Sales by Store Type','Store Type','Sales')

# ### Part 1.2 - Naive Clustering

# Use the Assortment feature as a natural cluster for naive clustering
stores_train['Assortment'].unique()

monthly_sales = stores_train.groupby(['Store',stores_train['Date'].dt.to_period('M')])['Sales'].mean().unstack(level=0)
monthly_sales = monthly_sales.set_index(monthly_sales.index.to_timestamp())
monthly_sales_a = stores_train[stores_train['Assortment']=='a']['Store'].unique()
monthly_sales_b = stores_train[stores_train['Assortment']=='b']['Store'].unique()
monthly_sales_c = stores_train[stores_train['Assortment']=='c']['Store'].unique()

len(monthly_sales_a)+len(monthly_sales_b)+len(monthly_sales_c)

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# #### Use the StoreType feature as a natural clustering for naive clustering
monthly_sales = stores_train.groupby(['Store',stores_train['Date'].dt.to_period('M')])['Sales'].mean().unstack(level=0)
monthly_sales = monthly_sales.set_index(monthly_sales.index.to_timestamp())
monthly_sales_a = stores_train[stores_train['StoreType']=='a']['Store'].unique()
monthly_sales_b = stores_train[stores_train['StoreType']=='b']['Store'].unique()
monthly_sales_c = stores_train[stores_train['StoreType']=='c']['Store'].unique()
monthly_sales_d = stores_train[stores_train['StoreType']=='d']['Store'].unique()

len(monthly_sales_a)+len(monthly_sales_b)+len(monthly_sales_c)+len(monthly_sales_d)

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='blue', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_d:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# #### Both StoreType and Assortment are not considered to be a good feature for clustering
# ### Part 1.3 - Time Series Clustering

#setting of training dates
train["Date"] = pd.to_datetime(train["Date"], infer_datetime_format = True)
exhaust_dates = pd.date_range(start=train["Date"].unique().min(), end=train["Date"].unique().max())
info_df = pd.DataFrame()

#checking the missing data and dates
for store in train["Store"].unique():
    cur_store_df = train[train.Store == store]
    uniq_date = cur_store_df["Date"].unique()
    no_uniq_date = len(uniq_date)
    missing_dates = exhaust_dates.difference(uniq_date)

    if len(missing_dates) == 0:
        missing_dates = ""
    else:
        missing_dates = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates]
        missing_dates = [date + " " for date in missing_dates]
        missing_dates = str(missing_dates)
    cur_info_df = pd.DataFrame({"Store":store, "no_unique_date":no_uniq_date, "missing_dates":missing_dates}, index = [0])
    info_df = info_df.append(cur_info_df, ignore_index = True)

#missing dates stores
missing_date_df = info_df[info_df.no_unique_date != len(train["Date"].unique())]
missing_date_stores = missing_date_df["Store"].values
print("Store with missing dates")
print(missing_date_stores)
print("\nValue counts of missing store")
print(missing_date_df["no_unique_date"].value_counts())
print("\nThe only one that does not have 758 missing")
print(missing_date_df[missing_date_df.no_unique_date != 758])


missing_date_df["missing_dates"][12]
missing_date_df["missing_dates"].value_counts()


#creation of new df to store
ts_train = pd.DataFrame({})
ts_train.index = exhaust_dates

#creation of a new train_dataset
temp_train = train.copy()
temp_train.index = temp_train.Date

#flattening
for store in range(1, temp_train["Store"].max()+1):
    ts_train[store] = temp_train[temp_train.Store==store]["Sales"]

ts_train.head()


#checking of ts_train for store 988 and the store sales on that day
print(ts_train.loc["2013-01-01",:].value_counts())
print("\n")
print(ts_train.loc["2013-01-01",ts_train.loc["2013-01-01",:] != 0])
non_na_20130101 = ts_train.loc["2013-01-01",ts_train.loc["2013-01-01",:] != 0].index.values
print("\n")
train[(train.Date == pd.to_datetime("2013-01-01")) & [(store in non_na_20130101) for store in train.Store]]

#assigning mode to 2013-01-01 for 988
ts_train.loc["2013-01-01", 988] = ts_train.loc["2013-01-01",:].mode().values[0]
ts_train[988]

#removing 988 from missing date store as it has been filled
missing_date_stores = list(np.delete(missing_date_stores, np.where(missing_date_stores == 988)))

#preepparing the ts dataset
ts_train_758 = []
index_758 = []
ts_train_941 =[]
index_941 = []

for store in range(1, temp_train["Store"].max()+1):
    if store in missing_date_stores:
        ts_train_758.append(ts_train[store].dropna().to_list())
        index_758.append(store)
    else:
        ts_train_941.append(ts_train[store].to_list())
        index_941.append(store)

#creating time series dataset and scaling
ts_train_758 = to_time_series_dataset(ts_train_758)
ts_train_758 = TimeSeriesScalerMeanVariance().fit_transform(ts_train_758)
print(ts_train_758.shape)


ts_train_941 = to_time_series_dataset(ts_train_941)
ts_train_941 = TimeSeriesScalerMeanVariance().fit_transform(ts_train_941)
print(ts_train_941.shape)

#fitting for 758
km = TimeSeriesKMeans(n_clusters=2, verbose=True, random_state=seed, metric = "dtw", max_iter = 10, n_jobs = -1)
y_pred_758 = km.fit_predict(ts_train_758)
sz_758 = ts_train_758.shape[1]

plt.figure(figsize = (20,10))
for yi in range(2):
    plt.subplot(2, 1, yi + 1)
    for xx in ts_train_758[y_pred_758 == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz_758)
    plt.ylim(-4, 4)
    plt.text(0.5, 1.02,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)

#fitting for 941
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed, max_iter = 10, metric = "dtw", n_jobs = -1)
y_pred_941 = km.fit_predict(ts_train_941)
sz_941 = ts_train_941.shape[1]


plt.figure(figsize = (20,15))
plt.title("DTW $k$-means for without missing dates")
for yi in range(3):
    plt.subplot(3, 1, yi + 1)
    for xx in ts_train_941[y_pred_941 == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz_941)
    plt.ylim(-4, 4)
    plt.text(0.5, 1.02,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)


#creation of dataframe to store the clusters
indexes = index_941 + index_758
y_pred_758_new = y_pred_758 + len(set(y_pred_941))
y_pred_all = np.append(y_pred_941, y_pred_758_new)
model_cluster = pd.DataFrame({"Store":indexes, "Cluster":y_pred_all})
print(model_cluster["Cluster"].value_counts().sort_index())
model_cluster.head()





#merge the cluster result to stores_train
stores_train_cluster = stores_train.merge(model_cluster, how = "left", on = "Store")
stores_train_cluster.head()





monthly_sales = stores_train_cluster.groupby(['Store',stores_train['Date'].dt.to_period('M')])['Sales'].mean().unstack(level=0)
monthly_sales = monthly_sales.set_index(monthly_sales.index.to_timestamp())
monthly_sales_a = stores_train_cluster[stores_train_cluster['Cluster']==0]['Store'].unique()
monthly_sales_b = stores_train_cluster[stores_train_cluster['Cluster']==1]['Store'].unique()
monthly_sales_c = stores_train_cluster[stores_train_cluster['Cluster']==2]['Store'].unique()
monthly_sales_d = stores_train_cluster[stores_train_cluster['Cluster']==3]['Store'].unique()
monthly_sales_e = stores_train_cluster[stores_train_cluster['Cluster']==4]['Store'].unique()


# #### Time Series Clustering Result




plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()





plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()





plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()





plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_d:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='blue', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()





plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_e:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='purple', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# #### The clusters of the missing records stores are similar, we should group them to one cluster




model_cluster['Cluster'] = model_cluster['Cluster'].replace(4,3)
model_cluster['Cluster'].head()


# #### Abnornal Patterns




abnormal= monthly_sales.min().to_frame()
zero_abnormal = abnormal[abnormal[0]==0].index





plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in zero_abnormal:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='black', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ## Part 2 - Data Preprocessing

# ### Part 2.1 - Stores df Preprocessing (non-cluster specfic)




#performing one hot encoding for store_type and Assortment (dropping A for both columns to preven multi-collinearity)
stores_processed = pd.get_dummies(stores, columns = ["StoreType","Assortment"], drop_first = True)
stores_processed.head()


#
# ### Part 2.2 - Merged Dataset Preprocessing (non-cluster-specific)

# #### Part 2.2.1 - Merging




stores_train_processed = train.merge(stores_processed, how = "left", on = "Store")
stores_train_processed.head()





stores_test_processed = test.merge(stores_processed, how = "left", on = "Store")
stores_test_processed.head()


# #### Part 2.2.2 - Prom2 Duration




stores_train_processed['Promo2SinceWeek'].fillna(-1, inplace=True) #fill the missing value with -1
stores_train_processed['Promo2SinceYear'].fillna(-1, inplace=True) #fill the missing value with -1
stores_train_processed['Promo2SinceWeek'] = stores_train_processed['Promo2SinceWeek'].astype(int) #convert to integer
stores_train_processed['Promo2SinceYear'] = stores_train_processed['Promo2SinceYear'].astype(int) #convert to integer

#compute the Promo2StartDate, assume Monday as 1st day in a week, return NaT (Not a Time) for missing values
stores_train_processed['Promo2StartDate'] = pd.to_datetime(stores_train_processed['Promo2SinceYear'].astype(str) + '-' +
                                                     stores_train_processed['Promo2SinceWeek'].astype(str) + '-' + '1', format='%Y-%W-%w'
                                                     ,errors='coerce')

#compute the Promo2Duration column
stores_train_processed['Promo2Duration'] = stores_train_processed['Date'] - stores_train_processed['Promo2StartDate']
stores_train_processed['Promo2Duration'] = stores_train_processed['Promo2Duration'].dt.days #convert from timedelta to float
stores_train_processed.loc[(stores_train_processed['Promo2Duration'] < 0)|(stores_train_processed['Promo2Duration'].isna()),
                           'Promo2Duration'] = 0 #change negative values and Nan to 0

#drop Promo2SinceWeek and Promo2SinceYear
stores_train_processed = stores_train_processed.drop(columns=['Promo2StartDate','Promo2SinceWeek','Promo2SinceYear'])
stores_train_processed.head()





stores_test_processed['Promo2SinceWeek'].fillna(-1, inplace=True) #fill the missing value with -1
stores_test_processed['Promo2SinceYear'].fillna(-1, inplace=True) #fill the missing value with -1
stores_test_processed['Promo2SinceWeek'] = stores_test_processed['Promo2SinceWeek'].astype(int) #convert to integer
stores_test_processed['Promo2SinceYear'] = stores_test_processed['Promo2SinceYear'].astype(int) #convert to integer

#compute the Promo2StartDate, assume Monday as 1st day in a week, return NaT (Not a Time) for missing values
stores_test_processed['Promo2StartDate'] = pd.to_datetime(stores_test_processed['Promo2SinceYear'].astype(str) + '-' +
                                                     stores_test_processed['Promo2SinceWeek'].astype(str) + '-' + '1', format='%Y-%W-%w'
                                                     ,errors='coerce')

#compute the Promo2Duration column
stores_test_processed['Promo2Duration'] = stores_test_processed['Date'] - stores_test_processed['Promo2StartDate']
stores_test_processed['Promo2Duration'] = stores_test_processed['Promo2Duration'].dt.days #convert from timedelta to float
stores_test_processed.loc[(stores_test_processed['Promo2Duration'] < 0)|(stores_test_processed['Promo2Duration'].isna()),
                           'Promo2Duration'] = 0 #change negative values and Nan to 0

#drop Promo2SinceWeek and Promo2SinceYear
stores_test_processed = stores_test_processed.drop(columns=['Promo2StartDate', 'Promo2SinceWeek','Promo2SinceYear'])
stores_test_processed.head()


# #### Part 2.2.3 - Competition Duration




stores_train_processed['CompetitionOpenSinceYear'].fillna(-1, inplace=True) #fill the missing value with -1
stores_train_processed['CompetitionOpenSinceMonth'].fillna(-1, inplace=True) #fill the missing value with -1
stores_train_processed['CompetitionOpenSinceYear'] = stores_train_processed['CompetitionOpenSinceYear'].astype(int) #convert to integer
stores_train_processed['CompetitionOpenSinceMonth'] = stores_train_processed['CompetitionOpenSinceMonth'].astype(int) #convert to integer

#compute the CompetitionOpenDate, return NaT (Not a Time) for missing values
stores_train_processed['CompetitionOpenDate'] = pd.to_datetime(stores_train_processed['CompetitionOpenSinceYear'].astype(str) + '-' +
                                                     stores_train_processed['CompetitionOpenSinceMonth'].astype(str).str.zfill(2),
                                                     errors='coerce')


#compute the CompetitionLength column
stores_train_processed['CompetitionLength'] = stores_train_processed['Date'] - stores_train_processed['CompetitionOpenDate']
stores_train_processed['CompetitionLength'] = stores_train_processed['CompetitionLength'].dt.days #convert from timedelta to float
stores_train_processed.loc[stores_train_processed['CompetitionLength'] < 0, 'CompetitionLength'] = 0 #change negative values to 0

#drop CompetitionOpenSinceMonth and CompetitionOpenSinceYear
stores_train_processed = stores_train_processed.drop(columns=['CompetitionOpenDate', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear'])
stores_train_processed.head()





stores_test_processed['CompetitionOpenSinceYear'].fillna(-1, inplace=True) #fill the missing value with -1
stores_test_processed['CompetitionOpenSinceMonth'].fillna(-1, inplace=True) #fill the missing value with -1
stores_test_processed['CompetitionOpenSinceYear'] = stores_test_processed['CompetitionOpenSinceYear'].astype(int) #convert to integer
stores_test_processed['CompetitionOpenSinceMonth'] = stores_test_processed['CompetitionOpenSinceMonth'].astype(int) #convert to integer

#compute the CompetitionOpenDate, return NaT (Not a Time) for missing values
stores_test_processed['CompetitionOpenDate'] = pd.to_datetime(stores_test_processed['CompetitionOpenSinceYear'].astype(str) + '-' +
                                                     stores_test_processed['CompetitionOpenSinceMonth'].astype(str).str.zfill(2),
                                                     errors='coerce')


#compute the CompetitionLength column
stores_test_processed['CompetitionLength'] = stores_test_processed['Date'] - stores_test_processed['CompetitionOpenDate']
stores_test_processed['CompetitionLength'] = stores_test_processed['CompetitionLength'].dt.days #convert from timedelta to float
stores_test_processed.loc[stores_test_processed['CompetitionLength'] < 0, 'CompetitionLength'] = 0 #change negative values to 0

#drop CompetitionOpenSinceMonth and CompetitionOpenSinceYear
stores_test_processed = stores_test_processed.drop(columns=['CompetitionOpenDate', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear'])
stores_test_processed.head()


# #### Part 2.2.4 - Prom Since Feature Engineering (ongoing month)




def Prom2_Interval(row):
    if pd.isnull(row["PromoInterval"]):
        return 0
    elif row["PromoInterval"] == "Jan,Apr,Jul,Oct":
        return 3-min([abs(x-row["Date"].month) for x in [1,4,7,10]])
    elif row["PromoInterval"] == "Feb,May,Aug,Nov":
        return 3-min([abs(x-row["Date"].month) for x in [2,5,8,11]])
    else:
        return 3-min([abs(x-row["Date"].month) for x in [3,6,9,12]])





stores_train_processed["Prom2_Ongoing"] = stores_train_processed.apply(Prom2_Interval, axis = 1)
stores_train_processed.drop(columns = ["PromoInterval"], inplace = True)
stores_train_processed.head()





stores_test_processed["Prom2_Ongoing"] = stores_test_processed.apply(Prom2_Interval, axis = 1)
stores_test_processed.drop(columns = ["PromoInterval"], inplace = True)
stores_test_processed.head()


# #### Part 2.2.5 - StateHoliday




stores_train_processed['StateHoliday'].replace(0,'0', inplace=True) #replace 0 by '0'
stores_test_processed['StateHoliday'].replace(0,'0', inplace=True) #replace 0 by '0'





stores_train_processed['StateHoliday'].value_counts()





stores_test_processed['StateHoliday'].value_counts()





#fitting encoder
sh_enc = OneHotEncoder(drop = "first")
sh_enc.fit(stores_train_processed[['StateHoliday']])

#estimating train and test
sh_enc_train = pd.DataFrame(sh_enc.transform(stores_train_processed[['StateHoliday']]).toarray(), columns = sh_enc.get_feature_names_out())
sh_enc_test = pd.DataFrame(sh_enc.transform(stores_test_processed[['StateHoliday']]).toarray(), columns = sh_enc.get_feature_names_out())

#merging
stores_train_processed = stores_train_processed.merge(sh_enc_train, left_index = True, right_index = True)
stores_train_processed.drop(columns = ["StateHoliday"], inplace = True)

stores_test_processed = stores_test_processed.merge(sh_enc_train, left_index = True, right_index = True)
stores_test_processed.drop(columns = ["StateHoliday"], inplace = True)





stores_train_processed.head()





stores_test_processed.head()


# #### Part 2.2.6 - Missing Value Checking




stores_train_processed.isna().sum()





stores_test_processed.isna().sum()


# ### Part 2.3 - Model Partitioning (Cluster)

# #### Part 2.3.1 - Creation of Datasets




cluster_a = stores_train_processed[stores_train_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 0]["Store"]))]
cluster_b = stores_train_processed[stores_train_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 1]["Store"]))]
cluster_c = stores_train_processed[stores_train_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 2]["Store"]))]
cluster_d = stores_train_processed[stores_train_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 3]["Store"]))]





cluster_a.head()





len(cluster_a) + len(cluster_b) + len(cluster_c) + len(cluster_d)





cluster_a_test = stores_test_processed[stores_test_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 0]["Store"]))]
cluster_b_test = stores_test_processed[stores_test_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 1]["Store"]))]
cluster_c_test = stores_test_processed[stores_test_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 2]["Store"]))]
cluster_d_test = stores_test_processed[stores_test_processed.Store.isin(list(model_cluster[model_cluster.Cluster == 3]["Store"]))]





cluster_a_test.head()





len(cluster_a_test)+len(cluster_b_test)+len(cluster_c_test)+len(cluster_d_test)


# #### Part 2.3.2 - Cluster Specific EDA




def cluster_boxplot(x,y,data,title,xlabel,ylabel, axis): #function for grouped boxplots
    box_plot = sns.boxplot(x=x, y=y, data=data, ax=axis)


    #modify individual font size of elements
    sns.set(rc = {'figure.figsize':(20, 7)})
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=17)





cluster_a_open = cluster_a[cluster_a['Open']==1]
fig, axes = plt.subplots(4, 1, figsize=(15,15))
cluster_boxplot('DayOfWeek', 'Sales', cluster_a_open, 'Distribution of sales per day of the week - Cluster a', 'Day of the week', 'Sales', axes[0])
cluster_boxplot(cluster_a_open['Date'].dt.day, 'Sales', cluster_a_open,'Distribution of sales per day of the month - Cluster a', 'Day of the month', 'Sales', axes[1])
cluster_boxplot(cluster_a_open['Date'].dt.month, 'Sales', cluster_a_open,'Distribution of sales per month - Cluster a', 'Month', 'Sales', axes[2])
cluster_boxplot(cluster_a_open['Date'].dt.weekofyear, 'Sales', cluster_a_open,'Distribution of sales per week - Cluster a', 'Week', 'Sales', axes[3])
fig.tight_layout()





cluster_b_open = cluster_b[cluster_b['Open']==1]
fig, axes = plt.subplots(4, 1, figsize=(15,15))
cluster_boxplot('DayOfWeek', 'Sales', cluster_b_open, 'Distribution of sales per day of the week - Cluster b', 'Day of the week', 'Sales', axes[0])
cluster_boxplot(cluster_b_open['Date'].dt.day, 'Sales', cluster_b_open,'Distribution of sales per day of the month - Cluster b', 'Day of the month', 'Sales', axes[1])
cluster_boxplot(cluster_b_open['Date'].dt.month, 'Sales', cluster_b_open,'Distribution of sales per month - Cluster b', 'Month', 'Sales', axes[2])
cluster_boxplot(cluster_b_open['Date'].dt.weekofyear, 'Sales', cluster_b_open,'Distribution of sales per week - Cluster b', 'Week', 'Sales', axes[3])
fig.tight_layout()





cluster_c_open = cluster_c[cluster_c['Open']==1]
fig, axes = plt.subplots(4, 1, figsize=(15,15))
cluster_boxplot('DayOfWeek', 'Sales', cluster_c_open, 'Distribution of sales per day of the week - Cluster c', 'Day of the week', 'Sales', axes[0])
cluster_boxplot(cluster_c_open['Date'].dt.day, 'Sales', cluster_c_open,'Distribution of sales per day of the month - Cluster c', 'Day of the month', 'Sales', axes[1])
cluster_boxplot(cluster_c_open['Date'].dt.month, 'Sales', cluster_c_open,'Distribution of sales per month - Cluster c', 'Month', 'Sales', axes[2])
cluster_boxplot(cluster_c_open['Date'].dt.weekofyear, 'Sales', cluster_c_open,'Distribution of sales per week - Cluster c', 'Week', 'Sales', axes[3])
fig.tight_layout()





cluster_d_open = cluster_d[cluster_d['Open']==1]
fig, axes = plt.subplots(4, 1, figsize=(15,15))
cluster_boxplot('DayOfWeek', 'Sales', cluster_d_open, 'Distribution of sales per day of the week - Cluster d', 'Day of the week', 'Sales', axes[0])
cluster_boxplot(cluster_d_open['Date'].dt.day, 'Sales', cluster_d_open,'Distribution of sales per day of the month - Cluster d', 'Day of the month', 'Sales', axes[1])
cluster_boxplot(cluster_d_open['Date'].dt.month, 'Sales', cluster_d_open,'Distribution of sales per month - Cluster d', 'Month', 'Sales', axes[2])
cluster_boxplot(cluster_d_open['Date'].dt.weekofyear, 'Sales', cluster_d_open,'Distribution of sales per week - Cluster d', 'Week', 'Sales', axes[3])
fig.tight_layout()


# #### Part 2.3.3 - Missing Data Handling - Competition Distance, Competition Length, Open




#helper function
def replace_missing_open(cur_df):
    open_replace = []
    open_fill = pd.DataFrame(cur_df.groupby(by = "Date")["Open"].agg(pd.Series.mode))
    open_fill = open_fill.reset_index()

    for row in cur_df.iterrows():
        if pd.isna(row[1]["Open"]):
            open_replace.append(open_fill[open_fill.Date == row[1]["Date"]]["Open"].values[0])
        else:
            open_replace.append(row[1]["Open"])

    return(open_replace)





#cluster_a
print("NAs in Competition length across train, validation and test ")
print(cluster_a['CompetitionLength'].isna().sum(), cluster_a_test['CompetitionLength'].isna().sum())
cluster_a.drop(columns = ['CompetitionLength'], inplace = True)
cluster_a_test.drop(columns = ['CompetitionLength'], inplace = True)

print("NAs in Competition length across train, validation and test ")
print(cluster_a['CompetitionDistance'].isna().sum(),cluster_a_test['CompetitionDistance'].isna().sum())





#checking for open
print(cluster_a_test["Open"].isna().sum())





#cluster_b
print("NAs in Competition length across train, validation and test ")
print(cluster_b['CompetitionLength'].isna().sum(),cluster_b_test['CompetitionLength'].isna().sum())
cluster_b.drop(columns = ['CompetitionLength'], inplace = True)
cluster_b_test.drop(columns = ['CompetitionLength'], inplace = True)

print("NAs in Competition length across train, validation and test ")
print(cluster_b['CompetitionDistance'].isna().sum(),cluster_b_test['CompetitionDistance'].isna().sum())
cluster_b.drop(columns = ['CompetitionDistance'], inplace = True)
cluster_b_test.drop(columns = ['CompetitionDistance'], inplace = True)





#checking for open
print(cluster_b_test["Open"].isna().sum())
cluster_b_test["Open"] = replace_missing_open(cluster_b_test)
print(cluster_b_test["Open"].isna().sum())





#cluster_c
print("NAs in Competition length across train, validation and test ")
print(cluster_c['CompetitionLength'].isna().sum(),cluster_c_test['CompetitionLength'].isna().sum())
cluster_c.drop(columns = ['CompetitionLength'], inplace = True)
cluster_c_test.drop(columns = ['CompetitionLength'], inplace = True)

print("NAs in Competition length across train, validation and test ")
print(cluster_c['CompetitionDistance'].isna().sum(),cluster_c_test['CompetitionDistance'].isna().sum())





#checking for open
cluster_c_test["Open"].isna().sum()





#cluster_d
print("NAs in Competition length across train, validation and test ")
print(cluster_d['CompetitionLength'].isna().sum(),cluster_d_test['CompetitionLength'].isna().sum())
cluster_d.drop(columns = ['CompetitionLength'], inplace = True)
cluster_d_test.drop(columns = ['CompetitionLength'], inplace = True)

print("NAs in Competition length across train, validation and test ")
print(cluster_d['CompetitionDistance'].isna().sum(),cluster_d_test['CompetitionDistance'].isna().sum())
cluster_d.drop(columns = ['CompetitionDistance'], inplace = True)
cluster_d_test.drop(columns = ['CompetitionDistance'], inplace = True)





#checking for open
cluster_d_test["Open"].isna().sum()


# #### Part 2.3.4 - One Hot Encoding - Day of the week and Month of Year Clusters




def manual_ohe(clusters_week, clusters_month, clusters_day_of_month, cluster):
    for i, c in enumerate(clusters_week):
        c = [ele -1 for ele in c] #as day of week start from 0 to 6
        cluster[f'Cluster_day_of_week_{i}'] = cluster['Date'].dt.dayofweek.isin(c).astype(int)

    for i, c in enumerate(clusters_month):
        cluster[f'Cluster_month_{i}'] = cluster['Date'].dt.month.isin(c).astype(int)

    for i, c in enumerate(clusters_day_of_month):
        cluster[f'Cluster_day_of_month_{i}'] = cluster['Date'].dt.day.isin(c).astype(int)

    return cluster





def cycle_ohe(cluster):
    cluster["month_sin"] = np.sin(2 * np.pi * cluster["month"]/12)
    cluster["month_cos"] = np.cos(2 * np.pi * cluster["month"]/12)
    cluster["doy_sin"] = np.sin(2 * np.pi * cluster["doy"]/7)
    cluster["doy_cos"] = np.cos(2 * np.pi * cluster["doy"]/7)
    cluster.drop(columns = ["month", "doy"], inplace = True)
    return(cluster)





def ohe_clusters(clusters_week, clusters_month, clusters_day_of_month, cluster_all, cluster_test ,all_or_not, cycle):
    cluster_all = cluster_all.reset_index(drop = True)
    cluster_test = cluster_test.reset_index(drop = True)

    if all_or_not == False:
        cluster_all = manual_ohe(clusters_week, clusters_month, clusters_day_of_month, cluster_all)
        cluster_test = manual_ohe(clusters_week, clusters_month, clusters_day_of_month, cluster_test)

    else:
        cluster_all["month"] = cluster_all.Date.dt.month
        cluster_all["dow"] = cluster_all.Date.dt.dayofweek+1

        cluster_test["month"] = cluster_test.Date.dt.month
        cluster_test["dow"] = cluster_test.Date.dt.dayofweek+1

        if cycle == True:
            cluster_all = cycle_ohe(cluster_all)
            cluster_test = cycle_ohe(cluster_test)

        else:
            #fitting encoder
            sh_enc = OneHotEncoder(drop = "first")
            sh_enc.fit(cluster_all[["dow", "month"]])

            #estimating train and test
            sh_enc_all = pd.DataFrame(sh_enc.transform(cluster_all[["dow", "month"]]).toarray(), columns = sh_enc.get_feature_names_out())
            sh_enc_test = pd.DataFrame(sh_enc.transform(cluster_test[["dow", "month"]]).toarray(), columns = sh_enc.get_feature_names_out())

            #merging
            cluster_all = cluster_all.merge(sh_enc_all, left_index = True, right_index = True)
            cluster_all.drop(columns = ["dow", "month"], inplace = True)

            cluster_test = cluster_test.merge(sh_enc_test, left_index = True, right_index = True)
            cluster_test.drop(columns = ["dow", "month"], inplace = True)

            for i, c in enumerate(clusters_day_of_month):
                cluster_all[f'Cluster_day_of_month_{i}'] = cluster_all['Date'].dt.day.isin(c).astype(int)
                cluster_test[f'Cluster_day_of_month_{i}'] = cluster_test['Date'].dt.day.isin(c).astype(int)

    return cluster_all, cluster_test





#setting of model parameters
all_or_not = True
cycle = False

clusters_week_a = [[1,7], [6]]
clusters_month_a = [[7,8,12]]
clusters_day_of_month_a = [[1,2,3,15,16,17,29,30,31]]

clusters_week_b = [[1,7], [6]]
clusters_month_b = [[12]]
clusters_day_of_month_b = [[1,2,3,15,16,17,29,30,31],[11,24]]

clusters_week_c = [[1], [7]]
clusters_month_c = [[12]]
clusters_day_of_month_c =[[1,2,16,17,29,30,31],[11,24]]

clusters_week_d = [[1], [7]]
clusters_month_d =  [[12]]
clusters_day_of_month_d = [[2,16,30]]





#applying on train and validation
cluster_a, cluster_a_test = ohe_clusters(clusters_week_a, clusters_month_a, clusters_day_of_month_a, cluster_a, cluster_a_test, all_or_not, cycle)
cluster_b, cluster_b_test = ohe_clusters(clusters_week_b, clusters_month_b, clusters_day_of_month_b, cluster_b, cluster_b_test, all_or_not, cycle)
cluster_c, cluster_c_test = ohe_clusters(clusters_week_c, clusters_month_c, clusters_day_of_month_c, cluster_c, cluster_c_test, all_or_not, cycle)
cluster_d, cluster_d_test = ohe_clusters(clusters_week_d, clusters_month_d, clusters_day_of_month_d, cluster_d, cluster_d_test, all_or_not, cycle)





cluster_a.head()





cluster_a_test.head()


# #### Part 2.3.5 - Splitting Train, Validation, Test




cluster_a_train = cluster_a[cluster_a.Date < pd.to_datetime("2015-01-01")]
cluster_a_validate = cluster_a[cluster_a.Date >= pd.to_datetime("2015-01-01")]
print(f"For cluster A {len(cluster_a_train), len(cluster_a_validate), len(cluster_a_test)}")

cluster_b_train = cluster_b[cluster_b.Date < pd.to_datetime("2015-01-01")]
cluster_b_validate = cluster_b[cluster_b.Date >= pd.to_datetime("2015-01-01")]
print(f"For cluster B {len(cluster_b_train), len(cluster_b_validate), len(cluster_b_test)}")

cluster_c_train = cluster_c[cluster_c.Date < pd.to_datetime("2015-01-01")]
cluster_c_validate = cluster_c[cluster_c.Date >= pd.to_datetime("2015-01-01")]
print(f"For cluster C {len(cluster_c_train), len(cluster_c_validate), len(cluster_c_test)}")

cluster_d_train = cluster_d[cluster_d.Date < pd.to_datetime("2015-01-01")]
cluster_d_validate = cluster_d[cluster_d.Date >= pd.to_datetime("2015-01-01")]
print(f"For cluster D {len(cluster_d_train), len(cluster_d_validate), len(cluster_d_test)}")

print(f"For overall set {len(cluster_a)+len(cluster_b)+len(cluster_c)+len(cluster_d)}")
print(f"For test set {len(cluster_a_test)+len(cluster_b_test)+len(cluster_c_test)+len(cluster_d_test)}")





cluster_a_train.head()





cluster_a_validate.head()





cluster_a_train.columns





cluster_a_test.columns


# #### Part 2.3.6 - Checking if there is only one unique value




def check_unique_columns(train, val, test, cluster):
    print(f"\ncheckin for cluster {cluster}")
    values = pd.DataFrame(train.apply(pd.Series.nunique))
    to_drop = values[values[0]==1].index.to_list()
    print(f"columns to drop {to_drop}")
    for col in to_drop:
        print(col)
        print(train[col].value_counts())

    if len(to_drop) != 0:
        train = train.drop(columns = to_drop)
        val = val.drop(columns = to_drop)
        test = test.drop(columns = to_drop)

    return(train, val, test)





cluster_a_train, cluster_a_validate, cluster_a_test = check_unique_columns(cluster_a_train, cluster_a_validate, cluster_a_test, "a")
cluster_b_train, cluster_b_validate, cluster_b_test = check_unique_columns(cluster_b_train, cluster_b_validate, cluster_b_test,"b")
cluster_c_train, cluster_c_validate, cluster_c_test = check_unique_columns(cluster_c_train, cluster_c_validate, cluster_c_test,"c")
cluster_d_train, cluster_d_validate, cluster_d_test = check_unique_columns(cluster_d_train, cluster_d_validate, cluster_d_test,"d")


# ## Part 3 - Feature Selection

# #### Part 3.1 - Checking




print("Cluster A columns allignment")
print((cluster_a_train.columns == cluster_a_validate.columns) & (cluster_a_validate.columns == cluster_a_test.columns))
print("Cluster B columns allignment")
print((cluster_b_train.columns == cluster_b_validate.columns) & (cluster_b_validate.columns == cluster_b_test.columns))
print("Cluster C columns allignment")
print((cluster_c_train.columns == cluster_c_validate.columns) & (cluster_c_validate.columns == cluster_c_test.columns))
print("Cluster D columns allignment")
print((cluster_d_train.columns == cluster_d_validate.columns) & (cluster_d_validate.columns == cluster_d_test.columns))


# #### Part 3.2 - VIF Backwards feature selection




def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)





def VIF_v2(cluster2, not_in_scope, numerical):
    vif_df = pd.DataFrame()
    cols = [_ for _ in cluster2.columns if _ not in not_in_scope]
    cluster2 = cluster2[cols]

    #applying standardisation on numerical columns
    for col in numerical:
        if col in cluster2.columns:
            cluster2[col] = (cluster2[col]-cluster2[col].mean())/cluster2[col].std()


    vif_df['variable'] = cluster2.columns

    #calculate VIF for each predictor variable
    vif_df['VIF'] = [variance_inflation_factor(cluster2.values, i) for i in range(cluster2.shape[1])]

    print('VIF before selection:',vif_df)

    while True:
        if sum(vif_df['VIF'] > 5) > 0:
            array = vif_df[vif_df['VIF'] > 5]['VIF'].values
            if np.isin(array,math.inf).any():
                    col_max_vif = vif_df[vif_df['VIF']==math.inf]['variable'][:1]
            else:
                    col_max_vif = vif_df[vif_df['VIF']==max(array)]['variable']

            cluster2.drop(col_max_vif.values[0],axis=1,inplace=True)
            vif_df = pd.DataFrame()
            vif_df['variable'] = cluster2.columns
            vif_df['VIF'] = [variance_inflation_factor(cluster2.values, i) for i in range(cluster2.shape[1])]
            print('\nColumn removed:',col_max_vif.values[0])
            print('---------------------')
            print('VIF after selection:',vif_df)

        else:
            break

    keep_col = list(cluster2.columns)
    keep_col.extend(not_in_scope)

    return(keep_col)





not_in_scope = ["DayOfWeek", "Customers", "Store", "Date", "Sales", "Open"]
numerical = ["CompetitionDistance","CompetitionLength", "Prom2Duration","Prom2_Ongoing"]





#cluster a
to_keep_a = VIF_v2(cluster_a_train, not_in_scope, numerical)
cluster_a_train_final = cluster_a_train[to_keep_a]
cluster_a_validate_final = cluster_a_validate[to_keep_a]
cluster_a_test_final = cluster_a_test[to_keep_a]





#cluster b
to_keep_b = VIF_v2(cluster_b_train, not_in_scope, numerical)
cluster_b_train_final = cluster_b_train[to_keep_b]
cluster_b_validate_final = cluster_b_validate[to_keep_b]
cluster_b_test_final = cluster_b_test[to_keep_b]





#cluster c
to_keep_c = VIF_v2(cluster_c_train, not_in_scope, numerical)
cluster_c_train_final = cluster_c_train[to_keep_c]
cluster_c_validate_final = cluster_c_validate[to_keep_c]
cluster_c_test_final = cluster_c_test[to_keep_c]





#cluster d
to_keep_d = VIF_v2(cluster_d_train, not_in_scope, numerical)
cluster_d_train_final = cluster_d_train[to_keep_d]
cluster_d_validate_final = cluster_d_validate[to_keep_d]
cluster_d_test_final = cluster_d_test[to_keep_d]


# ## Part 4 - Naive Model




def rmspe(y_true, y_pred):
    y_true = np.where(y_true==0, np.nan, y_true)
    y_pred = np.where(y_true==0, np.nan, y_pred)
    return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))*100





#naive model helper function
def naive_model_month(train_df, val_df):
    train_df["month"] = train_df["Date"].dt.month
    benchmark = pd.DataFrame(train_df[train_df.Open ==1].groupby(by = ["Store", "month"])["Sales"].mean()).reset_index()
    train_df.drop(columns = ["month"], inplace = True)

    prediction = val_df[["Store","Date","Sales","Open"]]
    prediction["month"] = val_df["Date"].dt.month
    prediction = prediction.merge(benchmark, on = ["Store", "month"])
    prediction.loc[prediction.Open == 0, "Sales_y"] = 0

    return(prediction["Sales_x"].values, prediction["Sales_y"].values)





#naive model helper function
def naive_model_week(train_df, val_df):
    train_df["week"] = train_df["Date"].dt.weekofyear
    benchmark = pd.DataFrame(train_df[train_df.Open ==1].groupby(by = ["Store", "week"])["Sales"].mean()).reset_index()
    train_df.drop(columns = ["week"], inplace = True)

    prediction = val_df[["Store","Date","Sales","Open"]]
    prediction["week"] = val_df["Date"].dt.weekofyear
    prediction = prediction.merge(benchmark, on = ["Store", "week"])
    prediction.loc[prediction.Open == 0, "Sales_y"] = 0
    return(prediction["Sales_x"].values, prediction["Sales_y"].values)





#naive model helper function
def naive_model_day(orig_train_df, val_df):
    train_df["doy"] = train_df["Date"].dt.dayofyear
    benchmark = pd.DataFrame(train_df[train_df.Open ==1].groupby(by = ["Store", "doy"])["Sales"].mean()).reset_index()
    train_df.drop(columns = ["doy"], inplace = True)

    prediction = val_df[["Store","Date","Sales", "Open"]]
    prediction["doy"] = val_df["Date"].dt.dayofyear
    prediction = prediction.merge(benchmark, on = ["Store", "doy"])
    prediction.loc[prediction.Open == 0, "Sales_y"] = 0
    return(prediction["Sales_x"].values, prediction["Sales_y"].values)





#overall dataset
train_df = stores_train_processed[stores_train_processed.Date < pd.to_datetime("2015-01-01")]
val_df = stores_train_processed[stores_train_processed.Date >= pd.to_datetime("2015-01-01")]
c_type = "overall"
y_true, y_pred = naive_model_month(train_df, val_df)
print(f"The naive model (month) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_week(train_df, val_df)
print(f"The naive model (week) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_day(train_df, val_df)
print(f"The naive model (day) for the{c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")





#cluster A
train_df = cluster_a_train
val_df = cluster_a_validate
c_type = "Cluster A"
y_true, y_pred = naive_model_month(train_df, val_df)
print(f"The naive model (month) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_week(train_df, val_df)
print(f"The naive model (week) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_day(train_df, val_df)
print(f"The naive model (day) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")





#cluster B
train_df = cluster_b_train
val_df = cluster_b_validate
c_type = "Cluster B"
y_true, y_pred = naive_model_month(train_df, val_df)
print(f"The naive model (month) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_week(train_df, val_df)
print(f"The naive model (week) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_day(train_df, val_df)
print(f"The naive model (day) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")





#cluster C
train_df = cluster_c_train
val_df = cluster_c_validate
c_type = "Cluster C"
y_true, y_pred = naive_model_month(train_df, val_df)
print(f"The naive model (month) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_week(train_df, val_df)
print(f"The naive model (week) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_day(train_df, val_df)
print(f"The naive model (day) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")





#cluster D
train_df = cluster_d_train
val_df = cluster_d_validate
c_type = "Cluster D"
y_true, y_pred = naive_model_month(train_df, val_df)
print(f"The naive model (month) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_week(train_df, val_df)
print(f"The naive model (week) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")

y_true, y_pred = naive_model_day(train_df, val_df)
print(f"The naive model (day) for the {c_type} dataset has RMSPE of: {rmspe(y_true, y_pred)}%")


# ## Part 5 - Robust Models
#benchmarking
def model_test(X_train, y_train, X_val, y_val):
    models = {}
    models["MLR"] = LinearRegression()
    models["DT"] = DecisionTreeRegressor(random_state = seed)
    models["RF"] = RandomForestRegressor(random_state = seed, n_jobs = -1)
    models["GB"] = GradientBoostingRegressor(random_state = seed)
    models["XGB"] = XGBRegressor(random_state = seed)

    for i, model_name in enumerate(models.keys()):
        print(f"for {model_name}: ")
        start = time.time()
        cur_model = models[model_name]
        cur_model.fit(X_train, y_train)
        y_pred_train = cur_model.predict(X_train)
        y_pred_val = cur_model.predict(X_val)
        print(f"Train Set rmspe: {rmspe(y_train, y_pred_train)}%")
        print(f"Validation Set rmspe: {rmspe(y_val, y_pred_val)}%")
        print(f"Train Set r2: {r2_score(y_train, y_pred_train)}")
        print(f"Validation Set r2: {r2_score(y_val, y_pred_val)}")
        end = time.time()
        print(f"the computational time is {end-start} ")
        print("\n")
        del cur_model



# ### Part 5.1 - Original
# #### Part 5.1.1 - Cluster A

#cluster_a - all (kept one hot encoding of doy and month directly)
cluster_a_train_X = cluster_a_train_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_a_validate_X = cluster_a_validate_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_a_test_X = cluster_a_test_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])

cluster_a_train_y = cluster_a_train_final["Sales"]
cluster_a_validate_y = cluster_a_validate_final["Sales"]

model_test(cluster_a_train_X, cluster_a_train_y, cluster_a_validate_X, cluster_a_validate_y)


# #### Part 5.1.2 - Cluster B
#cluster_b - all (kept one hot encoding of doy and month directly)
cluster_b_train_X = cluster_b_train_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_b_validate_X = cluster_b_validate_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_b_test_X = cluster_b_test_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])

cluster_b_train_y = cluster_b_train_final["Sales"]
cluster_b_validate_y = cluster_b_validate_final["Sales"]

model_test(cluster_b_train_X, cluster_b_train_y, cluster_b_validate_X, cluster_b_validate_y)


# #### Part 5.1.3 - Cluster C
#cluster_c - all (kept one hot encoding of doy and month directly)
cluster_c_train_X = cluster_c_train_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_c_validate_X = cluster_c_validate_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_c_test_X = cluster_c_test_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])

cluster_c_train_y = cluster_c_train_final["Sales"]
cluster_c_validate_y = cluster_c_validate_final["Sales"]

model_test(cluster_c_train_X, cluster_c_train_y, cluster_c_validate_X, cluster_c_validate_y)


# #### Part 5.1.4 - Cluster D
#cluster_d - all (kept one hot encoding of doy and month directly)
cluster_d_train_X = cluster_d_train_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_d_validate_X = cluster_d_validate_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])
cluster_d_test_X = cluster_d_test_final.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date"])

cluster_d_train_y = cluster_d_train_final["Sales"]
cluster_d_validate_y = cluster_d_validate_final["Sales"]

model_test(cluster_d_train_X, cluster_d_train_y, cluster_d_validate_X, cluster_d_validate_y)


# ### Part 5.2 - Stacking - Week
def add_sales_proxy_week(train, validate, test, to_keep):
    #Calculate week average on train
    train = train.copy(deep = True)
    validate = validate.copy(deep = True)
    test = test.copy(deep = True)

    train["week"] = train["Date"].dt.week
    week_average = pd.DataFrame(train[train.Open!=0].groupby(by = ["Store", "week"])["Sales"].mean()).reset_index()
    week_average.rename(columns={'Sales':'sales_proxy_week'}, inplace=True)

    to_keep.append('sales_proxy_week')

    train = train.merge(week_average, on = ["Store", "week"])
    train = train.drop(columns = ["week"])
    train.loc[train["Open"]==0,'sales_proxy_week'] = 0
    train = train[to_keep]

    validate["week"] = validate["Date"].dt.week
    validate = validate.merge(week_average, on = ["Store", "week"])
    validate = validate.drop(columns = ["week"])
    validate.loc[validate["Open"]==0,'sales_proxy_week'] = 0
    validate = validate[to_keep]

    test["week"] = test["Date"].dt.week
    test = test.merge(week_average, on = ["Store", "week"])
    test = test.drop(columns = ["week"])
    test.loc[test["Open"]==0,'sales_proxy_week'] = 0
    test = test[to_keep]

    return (train, validate, test)

#Estimate week average on train and add to validate and test
cluster_a_train_week, cluster_a_validate_week, cluster_a_test_week = add_sales_proxy_week(cluster_a_train, cluster_a_validate, cluster_a_test, to_keep_a)
cluster_b_train_week, cluster_b_validate_week, cluster_b_test_week = add_sales_proxy_week(cluster_b_train, cluster_b_validate, cluster_b_test, to_keep_b)
cluster_c_train_week, cluster_c_validate_week, cluster_c_test_week = add_sales_proxy_week(cluster_c_train, cluster_c_validate, cluster_c_test, to_keep_c)
cluster_d_train_week, cluster_d_validate_week, cluster_d_test_week = add_sales_proxy_week(cluster_d_train, cluster_d_validate, cluster_d_test, to_keep_d)


# #### Part 5.2.1 - Cluster A
#cluster_a - all (kept one hot encoding of day and month directly)
cluster_a_train_X = cluster_a_train_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_a_validate_X = cluster_a_validate_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_a_test_X = cluster_a_test_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_a_train_y = cluster_a_train_week["Sales"]
cluster_a_validate_y = cluster_a_validate_week["Sales"]

model_test(cluster_a_train_X, cluster_a_train_y, cluster_a_validate_X, cluster_a_validate_y)


# #### Part 5.2.2 - Cluster B
#cluster_b - all (kept one hot encoding of doy and month directly)
cluster_b_train_X = cluster_b_train_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_b_validate_X = cluster_b_validate_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_b_test_X = cluster_b_test_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_b_train_y = cluster_b_train_week["Sales"]
cluster_b_validate_y = cluster_b_validate_week["Sales"]

model_test(cluster_b_train_X, cluster_b_train_y, cluster_b_validate_X, cluster_b_validate_y)


# #### Part 5.2.3 - Cluster C
#cluster_c - all (kept one hot encoding of doy and month directly)
cluster_c_train_X = cluster_c_train_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_c_validate_X = cluster_c_validate_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_c_test_X = cluster_c_test_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_c_train_y = cluster_c_train_week["Sales"]
cluster_c_validate_y = cluster_c_validate_week["Sales"]

model_test(cluster_c_train_X, cluster_c_train_y, cluster_c_validate_X, cluster_c_validate_y)


# #### Part 5.2.4 - Cluster D
#cluster_d - all (kept one hot encoding of doy and month directly)
cluster_d_train_X = cluster_d_train_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_d_validate_X = cluster_d_validate_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_d_test_X = cluster_d_test_week.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_d_train_y = cluster_d_train_week["Sales"]
cluster_d_validate_y = cluster_d_validate_week["Sales"]

model_test(cluster_d_train_X, cluster_d_train_y, cluster_d_validate_X, cluster_d_validate_y)


# ### Part 5.3 - Stacking - Month
def add_sales_proxy_month(train, validate, test):
    train = train.copy(deep = True)
    validate = validate.copy(deep = True)
    test = test.copy(deep = True)

    #Calculate week average on train
    train["month"] = train["Date"].dt.month
    month_average = pd.DataFrame(train[train.Open!=0].groupby(by = ["Store", "month"])["Sales"].mean()).reset_index()
    month_average.rename(columns={'Sales':'sales_proxy_month'}, inplace=True)

    train = train.merge(month_average, on = ["Store", "month"])
    train = train.drop(columns = ["month"])
    train.loc[train["Open"]==0,'sales_proxy_month'] = 0

    validate["month"] = validate["Date"].dt.month
    validate = validate.merge(month_average, on = ["Store", "month"])
    validate = validate.drop(columns = ["month"])
    validate.loc[validate["Open"]==0,'sales_proxy_month'] = 0

    test["month"] = test["Date"].dt.month
    test = test.merge(month_average, on = ["Store", "month"])
    test = test.drop(columns = ["month"])
    test.loc[test["Open"]==0,'sales_proxy_month'] = 0

    return (train, validate, test)

#Estimate week average on train and add to validate and test
cluster_a_train_month, cluster_a_validate_month, cluster_a_test_month = add_sales_proxy_month(cluster_a_train, cluster_a_validate, cluster_a_test)
cluster_b_train_month, cluster_b_validate_month, cluster_b_test_month = add_sales_proxy_month(cluster_b_train, cluster_b_validate, cluster_b_test)
cluster_c_train_month, cluster_c_validate_month, cluster_c_test_month = add_sales_proxy_month(cluster_c_train, cluster_c_validate, cluster_c_test)
cluster_d_train_month, cluster_d_validate_month, cluster_d_test_month = add_sales_proxy_month(cluster_d_train, cluster_d_validate, cluster_d_test)


# #### Part 5.3.1 - Cluster A
#cluster_a - all (kept one hot encoding of doy and month directly)
cluster_a_train_X = cluster_a_train_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_a_validate_X = cluster_a_validate_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_a_test_X = cluster_a_test_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_a_train_y = cluster_a_train_month["Sales"]
cluster_a_validate_y = cluster_a_validate_month["Sales"]

model_test(cluster_a_train_X, cluster_a_train_y, cluster_a_validate_X, cluster_a_validate_y)


# #### Part 5.3.2 - Cluster B
#cluster_b - all (kept one hot encoding of doy and month directly)
cluster_b_train_X = cluster_b_train_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_b_validate_X = cluster_b_validate_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_b_test_X = cluster_b_test_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_b_train_y = cluster_b_train_month["Sales"]
cluster_b_validate_y = cluster_b_validate_month["Sales"]

model_test(cluster_b_train_X, cluster_b_train_y, cluster_b_validate_X, cluster_b_validate_y)


# #### Part 5.2.3 - Cluster C
#cluster_c - all (kept one hot encoding of doy and month directly)
cluster_c_train_X = cluster_c_train_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_c_validate_X = cluster_c_validate_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_c_test_X = cluster_c_test_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_c_train_y = cluster_c_train_month["Sales"]
cluster_c_validate_y = cluster_c_validate_month["Sales"]

model_test(cluster_c_train_X, cluster_c_train_y, cluster_c_validate_X, cluster_c_validate_y)


# #### Part 5.3.4 - Cluster D
#cluster_d - all (kept one hot encoding of doy and month directly)
cluster_d_train_X = cluster_d_train_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_d_validate_X = cluster_d_validate_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])
cluster_d_test_X = cluster_d_test_month.drop(columns = ["DayOfWeek", "Customers", "Sales", "Store", "Date", "Open"])

cluster_d_train_y = cluster_d_train_month["Sales"]
cluster_d_validate_y = cluster_d_validate_month["Sales"]

model_test(cluster_d_train_X, cluster_d_train_y, cluster_d_validate_X, cluster_d_validate_y)


# ### Part 5.4 - Final Models - Predictions
#benchmarking
def model_test_v2(X_train, y_train, X_val, y_val):
    models = {}
    models["RF"] = RandomForestRegressor(random_state = seed, n_jobs = -1)
    models["XGB"] = XGBRegressor(random_state = seed)

    estimators = [
        ('rf', RandomForestRegressor(random_state = seed, n_jobs = -2)),
        ('xgb', XGBRegressor(random_state = seed))
    ]
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator= LinearRegression()
    )
    models["stack"] = stack_reg

    return_models = []

    for model_name in models.keys():
        print(f"for {model_name}: ")
        start = time.time()
        cur_model = models[model_name]
        cur_model.fit(X_train, y_train)
        y_pred_train = cur_model.predict(X_train)
        y_pred_val = cur_model.predict(X_val)
        print(f"Train Set rmspe: {rmspe(y_train, y_pred_train)}%")
        print(f"Validation Set rmspe: {rmspe(y_val, y_pred_val)}%")
        print(f"Train Set r2: {r2_score(y_train, y_pred_train)}")
        print(f"Validation Set r2: {r2_score(y_val, y_pred_val)}")
        end = time.time()
        print(f"the computational time is {end-start} ")
        print("\n")
        return_models.append(cur_model)

    return(return_models)


# #### Part 5.4.1 - Cluster A
models_a = model_test_v2(cluster_a_train_X, cluster_a_train_y, cluster_a_validate_X, cluster_a_validate_y)

#selection of final model
final_model_a = models_a[2]
final_model_a


# #### Part 5.4.2 - Cluster B
models_b = model_test_v2(cluster_b_train_X, cluster_b_train_y, cluster_b_validate_X, cluster_b_validate_y)

#selection of final model
final_model_b = models_b[0]
final_model_b


# #### Part 5.4.3 - Cluster C
models_c = model_test_v2(cluster_c_train_X, cluster_c_train_y, cluster_c_validate_X, cluster_c_validate_y)

#selection of final model
final_model_c = models_c[2]
final_model_c


# #### Part 5.4.4 - Cluster D
models_d = model_test_v2(cluster_d_train_X, cluster_d_train_y, cluster_d_validate_X, cluster_d_validate_y)

#selection of final model
final_model_d = models_d[2]
final_model_d

#Retrieves dataframes with predictions
#cluster --> a,b,c,d
#split --> train, validate, test

def retrieve_predictions(cluster,split):


    clusters = {
            'a': (cluster_a_train_month,cluster_a_validate_month,cluster_a_test_month,cluster_a_train_X,cluster_a_validate_X,cluster_a_test_X, final_model_a),
            'b': (cluster_b_train_month,cluster_b_validate_month,cluster_b_test_month,cluster_b_train_X,cluster_b_validate_X,cluster_b_test_X, final_model_b),
            'c': (cluster_c_train_month,cluster_c_validate_month,cluster_c_test_month,cluster_c_train_X,cluster_c_validate_X,cluster_c_test_X, final_model_c),
            'd': (cluster_d_train_month,cluster_d_validate_month,cluster_d_test_month,cluster_d_train_X,cluster_d_validate_X,cluster_d_test_X, final_model_d)
            }

    cluster_train_month,cluster_validate_month,cluster_test_month,cluster_train_X,cluster_validate_X,cluster_test_X,final_model = clusters[cluster]


    split1 =  {
        'train':(cluster_train_X,cluster_train_month),
        'validate':(cluster_validate_X,cluster_validate_month),
        'test': (cluster_test_X,cluster_test_month)
        }

    cluster_X, cluster_month = split1[split]

    cluster_y_pred = final_model.predict(cluster_X)
    cluster_y_pred =pd.DataFrame(cluster_y_pred)
    cluster_y_pred['Date'] = cluster_month.Date
    cluster_y_pred['Store'] = cluster_month.Store
    cluster_y_pred = cluster_y_pred.rename(columns={0:'Pred'})

    return cluster_y_pred


#train - predictions
cluster_a_train_y_pred = retrieve_predictions('a','train')
cluster_b_train_y_pred = retrieve_predictions('b','train')
cluster_c_train_y_pred = retrieve_predictions('c','train')
cluster_d_train_y_pred = retrieve_predictions('d','train')


#validate - predictions
cluster_a_validate_y_pred = retrieve_predictions('a','validate')
cluster_b_validate_y_pred = retrieve_predictions('b','validate')
cluster_c_validate_y_pred = retrieve_predictions('c','validate')
cluster_d_validate_y_pred = retrieve_predictions('d','validate')


#test - predictions
cluster_a_test_y_pred = retrieve_predictions('a','test')
cluster_b_test_y_pred = retrieve_predictions('b','test')
cluster_c_test_y_pred = retrieve_predictions('c','test')
cluster_d_test_y_pred = retrieve_predictions('d','test')


# ### Part 5.5 - Final Visualisations
#function for visualizations
#cluster --> a,b,c,d
#stores -->[]
from matplotlib.lines import Line2D

def plot_sales_data2(cluster, stores):

    cluster_vars = {
        'a': (cluster_a_train, cluster_a_validate, cluster_a_train_y_pred, cluster_a_validate_y_pred, cluster_a_test_y_pred),
        'b': (cluster_b_train, cluster_b_validate, cluster_b_train_y_pred, cluster_b_validate_y_pred, cluster_b_test_y_pred),
        'c': (cluster_c_train, cluster_c_validate, cluster_c_train_y_pred, cluster_c_validate_y_pred, cluster_c_test_y_pred),
        'd': (cluster_d_train, cluster_d_validate, cluster_d_train_y_pred, cluster_d_validate_y_pred, cluster_d_test_y_pred)
    }

    if cluster not in cluster_vars:
        print(f"Invalid cluster: {cluster}")
        return

    # Unpack variables from the dictionary
    cluster_train, cluster_validate, cluster_train_y_pred, cluster_validate_y_pred, cluster_test_y_pred = cluster_vars[cluster]

    # Create subplots for each store
    fig, axs = plt.subplots(nrows=len(stores), ncols=1, figsize=(14, 10), gridspec_kw={'hspace': 0.7, 'left': 0.01})

    # Add x and y labels to the subplots
    for ax in axs.flat:
        ax.set(xlabel='Date', ylabel='Sales')

    # Add a title to the overall figure
    fig.suptitle(f'Sales Time Series for Stores {stores} - Cluster {cluster}')



    # Plot sales data for each store
    for i, store in enumerate(stores):

        # Get train and validate data for the store
        train_data = cluster_train[cluster_train['Store'] == store][['Date', 'Sales']]
        train_data = train_data.sort_values(by="Date", ascending=False)
        validate_data = cluster_validate[cluster_validate['Store'] == store][['Date', 'Sales']]
        validate_data = validate_data.sort_values(by="Date", ascending=False)

        #forecast on train
        forecast_train = cluster_train_y_pred[cluster_train_y_pred['Store']==store][['Date','Pred']]
        forecast_train = forecast_train.sort_values(by="Date", ascending=False)

        #forecast on validate
        forecast_validate = cluster_validate_y_pred[cluster_validate_y_pred['Store']==store][['Date','Pred']]
        forecast_validate = forecast_validate.sort_values(by="Date", ascending=False)

        #forecast on test
        forecast_test = cluster_test_y_pred[cluster_test_y_pred['Store']==store][['Date','Pred']]
        forecast_test = forecast_test.sort_values(by="Date", ascending=False)


        # Plot train, validate and test data on the corresponding subplot
        axs[i].plot(train_data.Date, train_data.Sales, marker='', color='black', linewidth=0.8, alpha=0.5)
        axs[i].plot(validate_data.Date, validate_data.Sales, marker='', color='black', linewidth=0.8, alpha=0.5)
        axs[i].plot(forecast_train.Date, forecast_train.Pred, marker='', color='blue', linewidth=0.8, alpha=0.3)
        axs[i].plot(forecast_validate.Date, forecast_validate.Pred, marker='', color='blue', linewidth=0.8, alpha=0.3)
        axs[i].plot(forecast_test.Date, forecast_test.Pred, marker='', color='red', linewidth=0.8, alpha=0.5)
        axs[i].set_title(f"Store {store}", y =1.1)


        # Add vertical lines to indicate train/validate/test split
        axs[i].axvline(x=cluster_train.Date.max(), color='k', linestyle='--',linewidth=0.9)
        axs[i].axvline(x=cluster_validate.Date.max(), color='k', linestyle='--',linewidth=0.9)

        # Add text above the vertical lines
        axs[i].text(x=cluster_train.Date.max() - pd.Timedelta(days=30), y=axs[i].get_ylim()[1] * 1.05, s='Train | Validation', fontsize=10)
        axs[i].text(x=cluster_validate.Date.max() - pd.Timedelta(days=30), y=axs[i].get_ylim()[1] * 1.05, s='Validation | Test', fontsize=10)

        # Add a legend to the subplot
        legend_elements = [    Line2D([0], [0], color='black', linewidth=0.8, alpha=0.5, label='Train & Validate Data'),
            Line2D([0], [0], color='blue', linewidth=0.8, alpha=0.3, label='Train & Validate Forecast'),
            Line2D([0], [0], color='red', linewidth=0.8, alpha=0.5, label='Test Forecast')
            ]

        axs[i].legend(handles=legend_elements, loc='upper left',fontsize=11,bbox_to_anchor=(0, 1.3))


# #### Part 5.5.1 - Cluster A
cluster_a_test.Store.unique()

#Visualization cluster A (on randomly selected stores)
plot_sales_data2('a',[122,948,1045])

# #### Part 5.5.2 - Cluster B
cluster_b_test.Store.unique()

#Visualization cluster B (on randomly selected stores)
plot_sales_data2('b',[238,992,844])

# #### Part 5.5.3 - Cluster C
cluster_c_test.Store.unique()

#Visualization cluster C (on randomly selected stores)
plot_sales_data2('c',[10,1105,1115])

# #### Part 5.5.4 - Cluster D
cluster_d_test.Store.unique()

#Visualization cluster D (on randomly selected stores)
plot_sales_data2('d',[571,952,1000])
