
#Make a model to predict the app rating

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import re as re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

#------------------------------------------------------------------------------------------------
#Load the data file using pandas. 
#------------------------------------------------------------------------------------------------
df= pd.read_csv('.\\1569582940_googleplaystore\\googleplaystore.csv')
print(df.shape)
print(pd.options.display.max_rows)
#------------------------------------------------------------------------------------------------
#Check for null values in the data. Get the number of null values for each column.
#------------------------------------------------------------------------------------------------
print(df.isnull().sum())

#------------------------------------------------------------------------------------------------
#Drop records with nulls in any of the columns.
#------------------------------------------------------------------------------------------------ 
df=df.dropna()

#------------------------------------------------------------------------------------------------
"""Variables seem to have incorrect type and inconsistent formatting. You need to fix them: 
Size column has sizes in Kb as well as Mb. To analyze, you’ll need to convert these to numeric.
Extract the numeric value from the column
Multiply the value by 1,000, if size is mentioned in Mb"""
#------------------------------------------------------------------------------------------------
def Mb_to_kb(size):
     if size.endswith('k'):
         return float(size[:-1])
     elif size.endswith('M'):
         return float(size[:-1])*1000
     else:
         return size

df['Size'] = df['Size'].replace(['Varies with device'],'Nan')
df['Size'] = df['Size'].apply(lambda x: Mb_to_kb(x))
#print(df['Size'])
df['Size'] = df['Size'].astype(float)
df['Size'] = df.groupby('Category')['Size'].transform(lambda x: x.fillna(x.mean()))
print(df['Size'])

#------------------------------------------------------------------------------------------------
#Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).
#------------------------------------------------------------------------------------------------
df['Reviews'] = df['Reviews'].astype(int)

#------------------------------------------------------------------------------------------------
"""Installs field is currently stored as string and has values like 1,000,000+. 
Treat 1,000,000+ as 1,000,000
remove ‘+’, ‘,’ from the field, convert it to integer"""
#--------------------------------------------------------------------------------

df['Installs'] =  df['Installs'].str.replace('\+|,', '',regex=True).astype(int)

#--------------------------------------------------------------------------------------
#Price field is a string and has $ symbol. Remove ‘$’ sign, and convert it to numeric.
#-------------------------------------------------------------------------------------
df['Price'] =  df['Price'].str.replace('\$', '',regex=True).astype(float)
print(len(df.index))

#------------------------------------------------------------------------------------------------------------------------------------------------
"""5. Sanity checks:
Average rating should be between 1 and 5 as only these values are allowed on the play store. Drop the rows that have a value outside this range.
Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them.
For free apps (type = “Free”), the price should not be >0. Drop any such rows."""
#-------------------------------------------------------------------------------------------------------------------------------------------------
print("*****Rating*****")
print(len(df[(df['Rating'] < 1) & (df['Rating'] > 5)]))    #Zero rows.Nothing to drop.
print("*****Rating*****")
print(df.shape)

print("*****Reviews*****")
df=df.loc[(~(df['Reviews']>df['Installs']))]        # 7 rows dropped
print("*****Reviews***** 7 rows dropped",df.shape)

print("*****Type*****")
print(len(df[(df['Type'] == 'Free') & (df['Price'] != 0)])) #Zero rows.Nothing to drop.
print("*****Type*****")
print("df.shape")

#------------------------------------------------------------------------------------------------------------------------------------
#5. Performing univariate analysis: 
#-------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15,8))
sns.boxplot(x=df['Price'],color='skyblue').set(title='Price Boxplot')
plt.show()
#Price for few apps are very high which is very unusual as apps are fee or very little cost is assosiated with app subscriptions

plt.figure(figsize=(15,8))
sns.boxplot(x=df['Reviews']).set(title='Reviews Boxplot')
plt.show()
#Some of the apps has very high reviews which will skew the prediction.

plt.figure(figsize=(15,8))
sns.histplot(data=df,x='Rating').set(title='Rating Histplot')
plt.show()
#Ratings plot is left skwed

plt.figure(figsize=(15,8))
sns.histplot(data=df,x='Size').set(title='Size Histplot')
plt.xlabel("Size in Kb")
plt.show()
#Size plot is right skwed.It has outliers

#------------------------------------------------------------------------------------------------------------------------------------------------------
"""6. Outlier treatment: 
Price: From the box plot, it seems like there are some apps with very high price. A price of $200 for an application on the Play Store is very high and suspicious!
Check out the records with very high price
Is 200 indeed a high price?
Drop these as most seem to be junk apps
Reviews: Very few apps have very high number of reviews. These are all star apps that don’t help with the analysis and, in fact, will skew it. 
Drop records having more than 2 million reviews."""
#---------------------------------------------------------------------------------------------------------------------------------------------------------

df=df.loc[~((df['Price'] >200) |  (df['Reviews'] >2000000 ) )].reset_index(drop=True) #Dropped 468 rows
print(df.shape)

#---------------------------------------------------------------------------------------------------------------------------------------------
"""Installs:  There seems to be some outliers in this field too. Apps having very high number of installs should be dropped from the analysis.
Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
Decide a threshold as cutoff for outlier and drop records having values more than that."""
#---------------------------------------------------------------------------------------------------------------------------------------------
print(df['Installs'].quantile(q=[0.1,0.25,0.5,0.7,0.9,0.95,0.99]))
min_threshold,max_threshold=df['Installs'].quantile(q=[0.05,0.95])
df=df[(df['Installs'] >min_threshold) & (df['Installs'] <max_threshold)]
print(df.shape)

#----------------------------------------------------------------------------------------------------------------------------------------------------
"""7. Bivariate analysis: Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating.
 Make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features."""
#----------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(15,8))
sns.regplot(x='Price', y='Rating',data=df).set(title="Rating vs. Price")
plt.show()
# Price is having no effect on ratings.

plt.figure(figsize=(15,8))
sns.regplot(x='Size', y='Rating',data=df).set(title="Rating vs. Size")
plt.show()
# Apps small in size has higher rating than apps bigger in size.

plt.figure(figsize=(15,8))
sns.regplot(x='Reviews', y='Rating',data=df).set(title="Rating vs. Reviews")
plt.show()
# Apps with lower reviews also have been rated higher. It shows that higher volumn of reviews does not gurantee higher ratings.

plt.figure(figsize=(15,8))
sns.boxplot(x='Content Rating', y='Rating',data=df).set(title="Rating vs. Content Rating")
plt.show()
# Adult only 18+ and Everyone 10+ has more likes than any other types.

plt.figure(figsize=(15,8))
sns.boxplot(y='Rating', x='Category',data=df).set(title="Rating vs. Category")
plt.xticks(fontsize=8,rotation='vertical')
#plt.xticks(rotation=90)
plt.show()
# Art&Design ]category has no outliers and Education category has the highest ratings among all other categories. 

#----------------------------------------------------------------------------------------------------------------------------------------------------
"""8. Data preprocessing
For the steps below, create a copy of the dataframe to make all the edits. Name it inp1.
Reviews and Install have some values that are still relatively very high. Before building a linear regression model, you need to reduce the skew. Apply log transformation (np.log1p) to Reviews and Installs.
Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not useful for our task.
Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical data, and all data should be numeric. 
Dummy encoding is one way to convert character fields to numeric. Name of dataframe should be inp2."""
#--------------------------------------------------------------------------------------------------------------------------------------------
inp1=df.copy()
#print(inp1)
inp1['Reviews']=np.log1p(inp1['Reviews'])
print(inp1)
inp1['Installs']=np.log1p(inp1['Installs'])

inp1=inp1.drop(columns=['App', 'Last Updated','Current Ver','Android Ver','Type'])
print(inp1)
inp2=pd.get_dummies(inp1, columns=['Category','Content Rating','Genres'],drop_first=True)


#-----------------------------------------------------------------------------------------------------------
#9. Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.
#-----------------------------------------------------------------------------------------------------
df_train, df_test = train_test_split(inp2, train_size=0.70, random_state=0)
print(df_train.shape, df_test.shape)

#-----------------------------------------------------------------------------------------------------------
#10. Separate the dataframes into X_train, y_train, X_test, and y_test.
#-----------------------------------------------------------------------------------------------------------
y_train=df_train.Rating
x_train=df_train.drop(['Rating'],axis=1)
y_test=df_test.Rating
x_test=df_test.drop(['Rating'],axis=1)
print(x_train.shape,x_test.shape)

#-----------------------------------------------------------------------------------------------------------
"""11 . Model building
         Use linear regression as the technique
         Report the R2 on the train set"""
#-----------------------------------------------------------------------------------------------------------
linear_reg= LinearRegression()
linear_reg.fit(x_train,y_train)
y_train_pred =linear_reg.predict(x_train)
print(y_train_pred)
r2= linear_reg.score(x_train,y_train)
#r2=r2_score(y_train, y_train_pred).round(decimals=2)
print("R2 on tarin set: ",r2)
#-----------------------------------------------------------------------------------------------------------
#12. Make predictions on test set and report R2.
#-----------------------------------------------------------------------------------------------------------
y_test_pred =linear_reg.predict(x_test)
print(y_test_pred)
r2_test= linear_reg.score(x_test,y_test)
#r2_test=r2_score(y_test, y_test_pred).round(decimals=2)
print("R2 on test set: ",r2_test)

#-----------------------------------------------------------------------------------------------------------
#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
#-----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15,8))
sns.regplot(x=y_test, y=y_test_pred,data=df).set(title="y_test vs y_test_pred")         
plt.show()
# It shows +ve linear relationship between actual and predicted test values.

dfLRmodel = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred,'Difference':y_test-y_test_pred})
print(dfLRmodel)

# Dataframe of actual and predicted Rating of a app.
# We can see that for some app the model predicts errors and for some zero errors.
"""     Actual  Predicted  Difference
1799     4.3   4.086282    0.213718
3262     3.9   4.111184   -0.211184
392      4.1   4.180821   -0.080821
3790     4.3   4.250857    0.049143
123      3.8   3.947892   -0.147892
...      ...        ...         ...
5028     4.3   3.814348    0.485652
6936     4.3   4.214964    0.085036
1864     4.7   4.452333    0.247667
3747     4.3   4.261066    0.038934
19       4.0   4.249827   -0.249827 """

mse = mean_squared_error(y_test, y_test_pred)
print("Mean Square Error : ",mse)

rmse = np.sqrt(mse.round(decimals=2))
print("Root Mean Square Error: ",rmse)

# R2 score is very low therefore we can say that this model is not a good fit. 
# Means square error is low which is acceptable.
# Above analysis suggests to visualize this dataset using different visualization model in order to determine best fit model for this dataset. 
# Comapare the model with the higher R-squared value and lowest MSE value as a better fit for the data. 
