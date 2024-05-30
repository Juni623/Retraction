# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import re


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#get file encoding by this function  result: utf-8
# import chardet    

# def find_encoding(fname):
#     r_file = open(fname, 'rb').read()
#     result = chardet.detect(r_file)
#     charenc = result['encoding']
#     return charenc

# your_file = 'retractions35215.csv'
# print(find_encoding(your_file))               #utf-8


df = pd.read_csv("D:\\CDU\\S3\\564\\Assessment\\A3\\Juni\\new\\retractions35215.csv", delimiter=',', encoding='utf-8')
pd.set_option('display.max_columns', None)  # Set to display all columns
pd.set_option('display.expand_frame_repr', False)  # Set not to fold data
pd.set_option('max_colwidth', None)  # Set the maximum column width to unlimited



# get column_names by running print(df.index)
# column_names = [
#     'Record ID',
#     'Title', 
#     'Subject', 
#     'Institution', 
#     'Journal', 
#     'Publisher', 
#     'Country', 
#     'Author', 
#     'URLS', 
#     'ArticleType', 
#     'RetractionDate', 
#     'RetractionDOI', 
#     'RetractionPubMedID', 
#     'OriginalPaperDate', 
#     'OriginalPaperDOI',
#     'OriginalPaperPubMedID',
#     'Reason', 
#     'Notes']



#Clean missing vlaues from the imported data

df = df.fillna('Unknown')



#Remove Duplicates

df.drop_duplicates(inplace=True)
duplicates = df.duplicated().sum()
missing_values_count = df.isnull().sum()

df.tail()

# clean df['Country'] for some articles that have multiple countries

df_exploded_for_country = df.copy()
df_exploded_for_country['Country'] = df_exploded_for_country['Country'].str.split(';')
df_exploded_for_country = df_exploded_for_country.explode('Country')



# #Retractions by Country

fig, ax = plt.subplots(figsize=(6, 6))
df_exploded_for_country["Country"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='g', ax=ax, title="Retractions by Country"
)
ax.set_xlabel("Number of Training Examples")
plt.show()

# clean df['Subject'] for some articles that have multiple subjects

df_exploded_for_subject = df.copy()
df_exploded_for_subject['Subject'] = df_exploded_for_subject['Subject'].str.split(';')
df_exploded_for_subject = df_exploded_for_subject.explode('Subject')

# Remove empty or whitespace-only strings
df_exploded_for_subject = df_exploded_for_subject[df_exploded_for_subject['Subject'].str.strip() != '']

# #Retractions by Subject

fig, ax = plt.subplots(figsize=(4, 4))
df_exploded_for_subject["Subject"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='r', ax=ax, title="Retractions by Subject"
)
ax.set_xlabel("Number of Training Examples")
plt.show()


#clean df['Journal'] for some journals that have time of publication and (brief name)


def clean_journal_name(journal):
    # clean the time of publication
    journal = re.sub(r'\b\d{4}\b|\b\d+(?:st|nd|rd|th)\b', '', journal)
    # clean the brief name
    journal = re.sub(r'\([^()]*\)', '', journal)
    # clean the whitespace
    return journal.strip()

# get the new cleaned journal names
df['Journal'] = df['Journal'].apply(clean_journal_name)



# #Retractions by Journal

fig, ax = plt.subplots(figsize=(4, 4))
df["Journal"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='orange', ax=ax, title="Retractions by Journal"
)

ax.set_xlabel("Number of Training Examples")
plt.show()

# clean df['Publisher']

def clean_publisher_name(publisher):
    # Remove content within parentheses
    publisher = re.sub(r'\([^()]*\)', '', publisher)
    # Treat strings like Wiley-Blackwell and Wiley as the same publisher
    if '-' in publisher:
        publisher = publisher.split('-')[0]
    return publisher.strip()

# get the new cleaned publisher names
df_exploded_for_publisher = df.copy()
df_exploded_for_publisher['Publisher'] = df_exploded_for_publisher['Publisher'].apply(clean_publisher_name)



# #Retractions by Publishers
# 
# Retractions by Journals and Publishers show how they are compromised with the integrity and quality of their publications.

fig, ax = plt.subplots(figsize=(4, 4))
df_exploded_for_publisher["Publisher"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='purple', ax=ax, title="Retractions by Publisher"
)
ax.set_xlabel("Number of Training Examples")
plt.show()

# clean Reasons


df_exploded_for_reason = df.copy()
df_exploded_for_reason['Reason'] = df_exploded_for_reason['Reason'].str.split(';')
df_exploded_for_reason = df_exploded_for_reason.explode('Reason')
# delete empty or whitespace-only strings
df_exploded_for_reason = df_exploded_for_reason[df_exploded_for_reason['Reason'].str.strip() != '']
# #Retraction Reasons:

fig, ax = plt.subplots(figsize=(4, 4))
df_exploded_for_reason["Reason"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='orange', ax=ax, title="Retractions by Reason"
)
ax.set_xlabel("Number of Training Examples")
plt.show()



# #Retraction Nature

df["RetractionNature"].value_counts().plot.barh(color=['blue', '#f5005a', 'cyan', 'magenta'], title='Retraction Nature');

# #EDA with Classical Approach

# Visualization Libraries 
# ------------------------------
import seaborn as sns
import matplotlib.pyplot as pltb


# Machine Learning Models 
# --------------------------------------------------------------------------------------------------
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Customize to Remove Warnings and Better Observation 🔧
# --------------------------------------------------------

from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
   # display(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T) #Commented since we have str on cat features

check_df(df)

# #Columns

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical, and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                dataframe
        cat_th: int, optional
                threshold value for variables that appear numeric but are categorical
        car_th: int, optional
                threshold value for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                Cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# #Numerical Columns

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")


# for col in num_cols:
#     num_summary(df, col, True)

# #Categorical Columns

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)
    print("")

df['DaysBetween'] = (pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y') -
                     pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y')).dt.days
df['DaysBetween'].fillna(df['DaysBetween'].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df[['DaysBetween']],
                                                    df['CitationCount'],
                                                    test_size=0.3, random_state=42)


## df['DaysBetween'] = (pd.to_datetime(df['RetractionDate'], dayfirst=True) -
##                                 pd.to_datetime(df['OriginalPaperDate'], dayfirst=True)).dt.days
## df['DaysBetween'].fillna(df['DaysBetween'].mean(), inplace=True) 
## X_train, X_test, y_train, y_test = train_test_split( df[['DaysBetween']], 
##                                                        df['CitationCount'], 
##                                                        test_size=0.3, random_state=42)

    
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}, R-squared: {r2}")

data_numeric_possible = df.select_dtypes(include=[np.number])
data_numeric_possible.fillna(0, inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_numeric_possible)

    
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance: {explained_variance}")

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.xlim(-1, 12)
plt.show()

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)

df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('KMeans')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.xlim(-1, 12)
plt.show()

from sklearn.decomposition import IncrementalPCA

dt = df[['Reason', 'Country', 'CitationCount']]

# Preprocessing categorical variables
encoder = OneHotEncoder()  # Using sparse output to save memory
encoded_data = encoder.fit_transform(dt[['Reason', 'Country']])
# No need to convert to dense DataFrame since most operations can handle sparse matrices

# Binning CitationCount into categories for classification using fixed bins
bins = [-1, 0, 10, 100, np.inf]
labels = range(len(bins)-1)
dt['CitationClass'] = pd.cut(dt['CitationCount'], bins=bins, labels=labels)

# Splitting the dataset
X = encoded_data  # Use sparse matrix directly
y = dt['CitationClass'].cat.codes  # Using category codes instead of labels to save space
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train.toarray(), y_train)  # Convert to array only when necessary
y_pred = nb_classifier.predict(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Incremental PCA
n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=100)  # Batch size can be adjusted
X_ipca = ipca.fit_transform(X.toarray())  # Apply transformation in a controlled manner
print(f"Explained Variance: {ipca.explained_variance_ratio_}")

# Visualizing PCA with classifications
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_ipca[:, 0], X_ipca[:, 1], c=y, alpha=0.5, cmap='viridis')
plt.title('Incremental PCA of Dataset with Naive Bayes Classifications')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)
plt.grid(True)
plt.show()

dt = df[['Country', 'CitationCount']]

# Preprocessing categorical variables
encoder = OneHotEncoder()  # Sparse output to save memory
encoded_data = encoder.fit_transform(dt[['Country']])

# Binning CitationCount into categories for classification using fixed bins
bins = [-1, 0, 10, 100, np.inf]
labels = range(len(bins)-1)
dt['CitationClass'] = pd.cut(dt['CitationCount'], bins=bins, labels=labels)

# Combining encoded data with CitationClass
X = encoded_data  # Features from one-hot encoding
y = dt['CitationClass'].cat.codes  # Target classes for stratification

# Cross-Validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Perform cross-validation using SGDClassifier
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SGD Classifier approximates SVM with hinge loss
    svm_classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3)
    svm_classifier.fit(X_train, y_train)  # No need to convert to dense array
    y_pred = svm_classifier.predict(X_test)

    # Calculate and collect accuracy for each fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Printing the results
print("Accuracies across folds:", accuracies)
print("Average accuracy:", np.mean(accuracies))

Q1 = df['CitationCount'].quantile(0.25)
Q3 = df['CitationCount'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifying outliers
outliers = df[(df['CitationCount'] < lower_bound) | (df['CitationCount'] > upper_bound)]
print("Number of outliers:", outliers.shape[0])

# Option 1: Remove outliers
df_no_outliers = df[(df['CitationCount'] >= lower_bound) & (df['CitationCount'] <= upper_bound)]

# Option 2: Cap outliers
df_capped = df.copy()
df_capped['CitationCount'] = df_capped['CitationCount'].clip(lower=lower_bound, upper=upper_bound)

# Displaying changes
print("Original Data Size:", df.shape[0])
print("Data Size after Removing Outliers:", df_no_outliers.shape[0])
print("Data Size with Capped Outliers:", df_capped.shape[0])

interp = df[['Record ID', 'RetractionPubMedID', 'OriginalPaperPubMedID']]


# def corr_map(interp, width=14, height=6, annot_kws=15):
#     mtx = np.triu(interp.corr())
#     f, ax = plt.subplots(figsize = (width,height))
#     sns.heatmap(interp.corr(),
#                 annot= True,
#                 fmt = ".2f",
#                 ax=ax,
#                 vmin = -1,
#                 vmax = 1,
#                 cmap = "summer",
#                 mask = mtx,
#                 linewidth = 0.4,
#                 linecolor = "black",
#                 cbar=False,
#                 annot_kws={"size": annot_kws})
#     plt.yticks(rotation=0,size=15)
#     plt.xticks(rotation=75,size=15)
#     plt.title('\nCorrelation of the Interpolated data\n', size = 20)
#     plt.show();




df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y')
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y')
df['DaysBetween'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days




#Daysbetween prediction
#df['DaysBetween']=(pd.to_datetime(df['RetractionDate']) - pd.to_datetime(df['OriginalPaperDate'])).dt.days
#df['RetractionDate'] = pd.to_datetime(df['RetractionDate'])

df['YearMonth'] = df['RetractionDate'].dt.to_period('M')

datebe = df[['YearMonth','DaysBetween']].groupby('YearMonth').mean()
datebe.index=datebe.index.to_timestamp()
datebe_values = datebe.values.flatten()

plt.figure(figsize=(10, 6))
plt.bar(datebe.index, datebe_values, width=20,color='skyblue', edgecolor='black', linewidth=1.2, align='center')
plt.xlabel('Date')
plt.ylabel('Average Date Difference (days)')
plt.title('Monthly Average Date Between Retractions')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#feature Cleaning 
columns_to_encode = ['Subject', 'Institution', 'Journal', 'Publisher', 'Country', 'Author', 'URLS', 'ArticleType', 'Paywalled']

def frequency_based_encoding(column, df, threshold=0.8):
    freq = df[column].value_counts(normalize=True)
    cumulative_freq = freq.cumsum()
    top_labels = cumulative_freq[cumulative_freq <= threshold].index
    df[column + '_Label'] = df[column].apply(lambda x: x if x in top_labels else 'Other')
    label_encoder = LabelEncoder()
    df[column + '_Label'] = label_encoder.fit_transform(df[column + '_Label'])
    return label_encoder

label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = frequency_based_encoding(column, df)
    
#DOI cleanning 
df['DOI_Is_Same'] = df.apply(lambda x: 1 if x['RetractionDOI'] == x['OriginalPaperDOI'] else 0, axis=1)

#RetractionPubMedID
df['MedID_Is_Same'] = df.apply(lambda x: 1 if x['RetractionPubMedID'] == x['RetractionPubMedID'] else 0, axis=1)

#RetractionPubMedID
df['MedID_Is_Same'] = df.apply(lambda x: 1 if x['RetractionPubMedID'] == x['RetractionPubMedID'] else 0, axis=1)


numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

corr_map(interp, width=20, height=10, annot_kws=8)
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


#scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RetractionDate', y='CitationCount', data=df, color='blue', edgecolor='w', s=100)
plt.title('Scatter Plot of Number of Citations vs. Years Since Publication')
plt.xlabel('Years Since Publication')
plt.ylabel('Number of Citations')
plt.show()

#linear regression
X=df[['Subject_Label', 'Institution_Label','Journal_Label', 
      'Publisher_Label', 'Country_Label', 'Author_Label', 'URLS_Label'
      , 'ArticleType_Label', 'Paywalled_Label','DOI_Is_Same','MedID_Is_Same','MedID_Is_Same']]
y=df['DaysBetween']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("Predicted values:", y_pred)
print("Actual values:", y_test.values)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
print("MAE: %.2f " % mae)
print("MSE: %.2f " % mse)
print("RMSE: %.2f " % rmse)
print("R2: %.2f " % r2)


# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=1000, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("Predicted values:", y_pred_rf)
print("Actual values:", y_test.values)

mae = metrics.mean_absolute_error(y_test, y_pred_rf)
mse = metrics.mean_squared_error(y_test, y_pred_rf)
rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
r2 = metrics.r2_score(y_test, y_pred_rf)
print("MAE: %.2f " % mae)
print("MSE: %.2f " % mse)
print("RMSE: %.2f " % rmse)
print("R2: %.2f " % r2)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='blue', edgecolor='k', alpha=0.7, label='Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# Distribution of Prediction Errors
errors = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='skyblue', label='Prediction Errors')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.legend()

plt.annotate('Prediction Error = Actual Value - Predicted Value', 
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='left', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

plt.show()
# GridSearchCV

param_grid = {
    'n_estimators': [100, 500, 1000]
    # 'max_depth': [4, 8,  12],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [ 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


print("Best parameters found: ", grid_search.best_params_)


best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)


mae = metrics.mean_absolute_error(y_test, y_pred_rf)
mse = metrics.mean_squared_error(y_test, y_pred_rf)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred_rf)

print("Predicted values:", y_pred_rf)
print("Actual values:", y_test.values)

print("MAE: %.2f " % mae)
print("MSE: %.2f " % mse)
print("RMSE: %.2f " % rmse)
print("R2: %.2f " % r2)