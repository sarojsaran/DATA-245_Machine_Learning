#importing all the packages required for analysis
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import svm
import sklearn.metrics as metrics
from sklearn import preprocessing
import sklearn.linear_model as linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import difflib as dff


## import the data fill
input_file = "./H1B_Cleaned_data.csv"
print(input_file)

## read the data file into a dataframe
dataframe1 = pd.read_table(input_file, encoding="ISO-8859-1", sep = ',')

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
print(len(dataframe1))

dataframe1.isnull().any()

dataframe1.dropna(subset=['EMPLOYER_NAME','SOC_NAME','JOB_TITLE','COUNTRY'], inplace=True)

print(len(dataframe1))

dataframe1.isnull().any()

print(dataframe1.describe())

# Plotting a graph case status against Number of petition
Status_petition_plot = dataframe1['CASE_STATUS'].value_counts().plot(title = 'Number of petitions vs Petition Case Status \n',kind = 'bar', fontsize=12, color = 'purple',linewidth=2.0, figsize = (8, 7))
Status_petition_plot.set_ylabel("Petition case status\n", fontsize=14)
Status_petition_plot.set_xlabel("\n # of petitions\n", fontsize=14)
# Plots the graph

for p in Status_petition_plot.patches:
    Status_petition_plot.annotate(str(p.get_height()), (p.get_x() * 1.0050, p.get_height() * 1.005))
plt.show()
# Display the table
print(dataframe1['CASE_STATUS'].value_counts())

# Creating a temprary dataframe for only Certified, Denied and rejected cases
Temp_dataframe = dataframe1.loc[dataframe1['CASE_STATUS'].isin(["CERTIFIED", "DENIED", "REJECTED"])]
#Temp_dataframe=Temp_dataframe.sort_values(by=['YEAR'], ascending=True)
print(Temp_dataframe.head(5))

#datatype conversion and upper case formatting
Temp_dataframe['YEAR'] = Temp_dataframe['YEAR'].astype(int)
Temp_dataframe['EMPLOYER_NAME'] = Temp_dataframe['EMPLOYER_NAME'].str.upper()
Temp_dataframe['SOC_NAME'] = Temp_dataframe['SOC_NAME'].str.upper()
Temp_dataframe['JOB_TITLE'] = Temp_dataframe['JOB_TITLE'].str.upper()
Temp_dataframe['FULL_TIME_POSITION'] = Temp_dataframe['FULL_TIME_POSITION'].str.upper()
print(Temp_dataframe.head(5))

plot_status_petitions = Temp_dataframe['CASE_STATUS'].value_counts().plot(title = 'CASE STATUS vs # OF PETITIONS \n ',kind = 'bar', color = 'purple', figsize = (8, 7))
plot_status_petitions.set_xlabel("\n CASE STATUS")
plot_status_petitions.set_ylabel("NUMBER OF PETITIONS \n")
for p in plot_status_petitions.patches:
    plot_status_petitions.annotate(str(p.get_height()), (p.get_x() * 1.0050, p.get_height() * 1.000))
plot_status_petitions

print(" Top 10 Employers filing for petition \n ")
Top_employers_filing_for_petition= Temp_dataframe['EMPLOYER_NAME'].value_counts().head(10).plot.barh(title = "Top 10 Employers Filing for Petition \n", \
                                                                 fontsize=12, color = 'purple',linewidth=2.0, figsize = (7, 5))
Top_employers_filing_for_petition.set_ylabel("Employers \n",fontsize=13)
Top_employers_filing_for_petition.set_xlabel("\n Number of petitions",fontsize=12)
Top_employers_filing_for_petition
print(dataframe1['EMPLOYER_NAME'].value_counts().head(10))

print("Top 10 Positions in demand for H1B \n")
top10_occupation_demading_for_H1B= Temp_dataframe['SOC_NAME'].value_counts().head(15).plot.barh(title = "Top 10 positions in Demand \n", \
                                                                 fontsize=12, color = 'purple',linewidth=2.0, figsize = (7, 5))
top10_occupation_demading_for_H1B.set_ylabel("Occupation Name\n ",fontsize=12)
top10_occupation_demading_for_H1B.set_xlabel("\n Number of Petitions",fontsize=12)
top10_occupation_demading_for_H1B
print(dataframe1['SOC_NAME'].value_counts().head(15))

# Number of petitions evry year
#without sorting
yearwise_petition_plot=Temp_dataframe['YEAR'].value_counts().plot(title = "H1-B Petitions per year \n",\
                                                                kind='bar', fontsize=12, color = 'purple',linewidth=5,figsize = (9,8))
yearwise_petition_plot.set_xlabel('\n FILING YEAR ')
yearwise_petition_plot.set_ylabel(' NUMBER OF PETITIONS\n')
yearwise_petition_plot.tick_params(axis='x', which='major', labelsize=5)

                                  
#petition acceptance ratio per year
yearwise_acceptance_plot = pd.DataFrame(Temp_dataframe[Temp_dataframe['CASE_STATUS'] == 'CERTIFIED'].YEAR.value_counts() / Temp_dataframe.YEAR.value_counts())
yearwise_acceptance_plot = yearwise_acceptance_plot.sort_values(['YEAR'])
yearwise_acceptance_plot = yearwise_acceptance_plot.plot(title = 'Yearwise H1-B Petitions acceptance \n', kind = 'line', fontsize=12, color = 'purple',linewidth=5, figsize = (9, 8))
yearwise_acceptance_plot.set_xlabel('\nFILING YEAR')
yearwise_acceptance_plot.set_ylabel('Petition ACCEPTANCE RATIO\n')
yearwise_petition_plot.tick_params(axis='x', which='major', labelsize=10)
plt.show()

#petition acceptance ratio per year
yearwise_rejection_plot = pd.DataFrame(Temp_dataframe[Temp_dataframe['CASE_STATUS'] != 'CERTIFIED'].YEAR.value_counts() / Temp_dataframe.YEAR.value_counts())
yearwise_rejection_plot = yearwise_rejection_plot.sort_values(['YEAR'])
yearwise_rejection_plot = yearwise_rejection_plot.plot(title = 'Yearwise H1-B Petitions rejected/denied \n', kind = 'line', fontsize=12, color = 'purple',linewidth=5, figsize = (9, 8))
yearwise_rejection_plot.set_xlabel('\nFILING YEAR')
yearwise_rejection_plot.set_ylabel('Petition rejected/denied RATIO\n')
yearwise_rejection_plot.tick_params(axis='x', which='major', labelsize=10)
plt.show()

# Median salary per year
Salary_vs_year = Temp_dataframe.loc[:,['PREVAILING_WAGE', 'YEAR']].groupby(['YEAR']).agg(['median'])

Salary_vs_year = Salary_vs_year.plot(title='Yearwise Median Salary of employees\n',kind = 'bar', color = 'purple', legend = None, figsize = (8, 7))
Salary_vs_year.set_xlabel('\n PETITION FILING YEAR')
Salary_vs_year.set_ylabel('MEDiAN SALARY OF EMPLOYEES\n')
plt.show()
print(Salary_vs_year)

print(Temp_dataframe['CASE_STATUS'].unique())
Temp_dataframe = Temp_dataframe.loc[Temp_dataframe['CASE_STATUS'].isin(["CERTIFIED", "DENIED"])] #filtering
print(Temp_dataframe['CASE_STATUS'].unique())

Temp_dataframe.isnull().sum(axis = 0)

Temp_dataframe1 = Temp_dataframe
print(Temp_dataframe.shape)
print(Temp_dataframe1.shape)

print(Temp_dataframe1.CASE_STATUS.value_counts())

Temp_dataframe1_denied = Temp_dataframe1[Temp_dataframe1['CASE_STATUS'] == 'DENIED']

Temp_dataframe1_certified = Temp_dataframe1[Temp_dataframe1['CASE_STATUS'] == 'CERTIFIED']

Input_Certified, Input_Certified_extra, y_certified, y_certified_extra = \
train_test_split(Temp_dataframe1[Temp_dataframe1.CASE_STATUS == 'CERTIFIED'],Temp_dataframe1_certified.CASE_STATUS, train_size= 0.06, random_state=1)

training_df = Input_Certified.append(Temp_dataframe1_denied)

## plot the distribution of the certified and denied samples after downsampling
plot_after_ds = training_df['CASE_STATUS'].value_counts().plot(title = 'CASE STATUS vs NUMBER OF PETITIONS', \
                                                                kind = 'bar', color = 'purple')
plot_after_ds.set_xlabel("CASE STATUS")
plot_after_ds.set_ylabel("NUMBER OF PETITIONS")
for p in plot_after_ds.patches:
    plot_after_ds.annotate(str(p.get_height()), (p.get_x() * 1.0050, p.get_height() * 1.005))
plt.show()

# one-hot encoding for every possible and needed column
print("Dataframe with confirmed or denied cases :\n ")
print("*******************************************")
print(training_df.info())
print("*******************************************")
print("Unique values count of each columns :\n")
print("Case Status ",training_df.CASE_STATUS.nunique())
print("Unique Employers ",training_df.EMPLOYER_NAME.nunique())
print("Unique SOCs ", training_df.SOC_NAME.nunique())
print("Unique Job Titles ",training_df.JOB_TITLE.nunique())
print("Unique Employment Type ", training_df.FULL_TIME_POSITION.nunique())
print("Prevailing Wages ",training_df.PREVAILING_WAGE.nunique())
print("Unique Year ",training_df.YEAR.nunique())
print("Unique Worksite State ",training_df.WORKSITE.nunique())
print("Unique CITY State ",training_df.CITY.nunique())
print("Unique COUNTRY State ",training_df.COUNTRY.nunique())

def Prevailing_wage_categorization(prevailing_wage):
    if prevailing_wage <=50000:
        return "VERY LOW"
    elif prevailing_wage >50000 and prevailing_wage <= 70000:
        return "LOW"
    elif prevailing_wage >70000 and prevailing_wage <= 90000:
        return "MEDIUM"
    elif prevailing_wage >90000 and prevailing_wage<=150000:
        return "HIGH"
    elif prevailing_wage >=150000:
        return "VERY HIGH"
    
def Grant_status_Categorization(acceptance_ratio):
    if acceptance_ratio == -1:
        return "AR"
    elif acceptance_ratio >=0.0 and acceptance_ratio<0.20:
        return "VLA"
    elif acceptance_ratio>=0.20 and acceptance_ratio<0.40:
        return "LA"
    elif acceptance_ratio>=0.40 and acceptance_ratio<0.60:
        return "MA"
    elif acceptance_ratio>=0.60 and acceptance_ratio<0.80:
        return "HA"
    elif acceptance_ratio>=0.80:
        return "VHA"
def worksite_collector(worksite):
    return worksite.split(', ')[1]

training_df['WORKSITE'] = training_df['WORKSITE'].apply(worksite_collector)
training_df.WORKSITE.unique()

training_df['PREVAILING_WAGE_CATEGORY'] = training_df['PREVAILING_WAGE'].apply(Prevailing_wage_categorization)
print(training_df['PREVAILING_WAGE_CATEGORY'])

employer_tdf = training_df.loc[:,['EMPLOYER_NAME', 'CASE_STATUS']]
soc_tdf = training_df.loc[:,['SOC_NAME', 'CASE_STATUS']]
job_tdf = training_df.loc[:,['JOB_TITLE', 'CASE_STATUS']]
print(employer_tdf)

certified_employer_tdf = employer_tdf[employer_tdf.CASE_STATUS == 'CERTIFIED'].EMPLOYER_NAME
certified_soc_tdf = soc_tdf[soc_tdf.CASE_STATUS == 'CERTIFIED'].SOC_NAME
certified_job_tdf = job_tdf[job_tdf.CASE_STATUS == 'CERTIFIED'].JOB_TITLE
confirmed_employer_count = certified_employer_tdf.value_counts()
confirmed_SOC_count = certified_soc_tdf.value_counts()
confirmed_job_count = certified_job_tdf.value_counts()

total_employer_counts = employer_tdf.EMPLOYER_NAME.value_counts()
total_soc_counts = soc_tdf.SOC_NAME.value_counts()
total_job_counts = job_tdf.JOB_TITLE.value_counts()

ratio_final = confirmed_employer_count / total_employer_counts
ratio_final.fillna(-1, inplace=True)
employer_classification_final = ratio_final.apply(Grant_status_Categorization)
training_df['EMPLOYER_ACCEPTANCE'] = training_df.EMPLOYER_NAME.map(employer_classification_final)

ratio_final_soc = confirmed_SOC_count / total_soc_counts
ratio_final_soc.fillna(-1, inplace=True)
soc_classification_final = ratio_final_soc.apply(Grant_status_Categorization)
training_df['SOC_ACCEPTANCE'] = training_df.SOC_NAME.map(soc_classification_final)

ratio_final_job = confirmed_job_count / total_job_counts
ratio_final_job.fillna(-1, inplace=True)
job_classification_final = ratio_final_job.apply(Grant_status_Categorization)
training_df['JOB_ACCEPTANCE'] = training_df.JOB_TITLE.map(job_classification_final)

print("*******************************************")
print("Unique values count of each columns :\n")
print("Case Status ",training_df.CASE_STATUS.nunique())
print("Unique Employers ",training_df.EMPLOYER_ACCEPTANCE.nunique())
print("Wages Category", training_df.PREVAILING_WAGE_CATEGORY.nunique())
print("Unique SOCs ", training_df.SOC_ACCEPTANCE.nunique())
print("Unique Job Titles ",training_df.JOB_ACCEPTANCE.nunique())
print("Unique Filing Year ",training_df.YEAR.nunique())
print("Unique Worksite State ",training_df.WORKSITE.nunique())
print("Unique Employment Type ", training_df.FULL_TIME_POSITION.nunique())
print("*******************************************")
print(training_df.info())

case_status_dict = {"CERTIFIED" : 1, "DENIED": 0}
full_time_position_dict = {"Y" : 1, "N" : 0}
try:    
    training_df['CASE_STATUS'] = training_df['CASE_STATUS'].apply(lambda x: case_status_dict[x])
    training_df['FULL_TIME_POSITION'] = training_df['FULL_TIME_POSITION'].apply(lambda x: full_time_position_dict[x])
except:
    pass

training_df['YEAR'] = training_df['YEAR'].astype('int')
training_df.sort_index(inplace = True)
training_df = training_df.loc[:, ['CASE_STATUS', 'YEAR','WORKSITE', 'PREVAILING_WAGE_CATEGORY','EMPLOYER_ACCEPTANCE',\
                                  'JOB_ACCEPTANCE', 'SOC_ACCEPTANCE', 'FULL_TIME_POSITION']]
print(training_df.head())

final_dataframe_training = pd.get_dummies(training_df, columns=['YEAR', 'WORKSITE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE_CATEGORY', 'EMPLOYER_ACCEPTANCE',
                                                             
                                                                'JOB_ACCEPTANCE', 'SOC_ACCEPTANCE' ], drop_first=True)
print(final_dataframe_training.head())

classifer_model = LogisticRegression()
lr_rfe = RFE(classifer_model, 30)
fit = lr_rfe.fit(final_dataframe_training.iloc[:,1:], final_dataframe_training.iloc[:,0])
supporting_rfe = lr_rfe.support_
column_length = list(final_dataframe_training.iloc[:,1:].columns.values)
listing_selected = []
for index in range(len(column_length)):
    if supporting_rfe[index] == True:
        listing_selected.append(column_length[index])
    else:
        pass
print(listing_selected)
print(lr_rfe.ranking_)   

# ref.ranking_ is returning the array with all positive integer values which is indicating
#the attribute of rank of lower score and indicating a ranking of higjer value.

list_cols_unique = [col.split('_')[0] for col in listing_selected]
print(set(list_cols_unique))

x_train, x_test, Y_train, Y_test = train_test_split(final_dataframe_training.iloc[:,1:], final_dataframe_training.iloc[:, 0], test_size = 0.20, random_state=1)
#y_train[y_train==1].shape
print(Y_test[Y_test==1].shape)

x_train.head()

logistic_classification = linear_model.LogisticRegression()
logistic_classification.fit(x_train, Y_train) 

import pickle
pickle.dump(logistic_classification,open('logistic_classification.pickle', 'wb'))
with open('logistic_classification.pickle', 'rb') as f:
    lr = pickle.load(f)

print(open('logistic_classification.pickle', 'rb').read()[:40])

#pickled logistic regression prediction
lgr = lr.predict(x_test)
print(lgr)
probability = lr.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", lgr[:10])

print(metrics.confusion_matrix(Y_test,lgr))
print(metrics.classification_report(Y_test,lgr))

Y_prediction_lr = logistic_classification.predict(x_test)

probability = logistic_classification.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", Y_prediction_lr[:10])

print(metrics.confusion_matrix(Y_test,Y_prediction_lr))
print(metrics.classification_report(Y_test, Y_prediction_lr))

from sklearn import metrics
print("****For Logistic Regression****")
print()
print('\033[1m'+"Accuracy Score:"+ '\033[0m',metrics.accuracy_score(Y_test, Y_prediction_lr)) 
print('\033[1m'+"F1 Score:"+ '\033[0m',metrics.f1_score(Y_test, Y_prediction_lr))
print('\033[1m'+"Precision Score:"+ '\033[0m',metrics.precision_score(Y_test, Y_prediction_lr))
print('\033[1m'+"Recall Score:"+ '\033[0m',metrics.recall_score(Y_test, Y_prediction_lr))

from sklearn.metrics import roc_auc_score, roc_curve
Y_pred_proba = logistic_classification.predict_proba(x_test)
Y_pred_proba = Y_pred_proba[:, 1]
auc = roc_auc_score(Y_test, Y_pred_proba)
print('\033[1m'+'AUC:'+ '\033[0m' ' %.2f%%' % (auc*100))
print('\033[1m'+"Kappa Score:"+ '\033[0m',metrics.cohen_kappa_score(Y_test, Y_prediction_lr))
print('\033[1m'+"Mean Absolute Error:"+ '\033[0m',metrics.mean_absolute_error(Y_test, Y_prediction_lr))
print('\033[1m'+"Mean Squared Error:"+ '\033[0m',metrics.mean_squared_error(Y_test, Y_prediction_lr))
print('\033[1m'+"Root Mean Squared Error:"+ '\033[0m',np.sqrt(metrics.mean_squared_error(Y_test, Y_prediction_lr)))

Y_pred_proba = logistic_classification.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred_proba)

#create ROC curve
plt.title("ROC curve for Logistic Regression")
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

random_forest = RandomForestClassifier(n_estimators = 75, random_state = 50)
random_forest.fit(x_train, Y_train)

pickle.dump(random_forest,open('random_forest.pickle', 'wb'))
with open('random_forest.pickle', 'rb') as rf:
    randf = pickle.load(rf)

print(open('random_forest.pickle', 'rb').read()[:40])

#pickled random forest prediction
rfst = randf.predict(x_test)
print(rfst)
probability = randf.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", rfst[:10])

print(metrics.confusion_matrix(Y_test,rfst))
print(metrics.classification_report(Y_test,rfst))

Y_prediction_rf =  random_forest.predict(x_test)
probability_rf = random_forest.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", Y_prediction_rf[:10])
print(metrics.confusion_matrix(Y_test,Y_prediction_rf))
print(metrics.classification_report(Y_test, Y_prediction_rf))

from sklearn import metrics
print("****For Random Forest Classifier****")
print()
print('\033[1m'+"Accuracy Score:"+ '\033[0m',metrics.accuracy_score(Y_test, Y_prediction_rf)) 
print('\033[1m'+"F1 Score:"+ '\033[0m',metrics.f1_score(Y_test, Y_prediction_rf))
print('\033[1m'+"Precision Score:"+ '\033[0m',metrics.precision_score(Y_test, Y_prediction_rf))
print('\033[1m'+"Recall Score:"+ '\033[0m',metrics.recall_score(Y_test, Y_prediction_rf))

from sklearn.metrics import roc_auc_score, roc_curve
probs = random_forest.predict_proba(x_test)
probs = probs[:, 1]
auc = roc_auc_score(Y_test, probs)
print('\033[1m'+'AUC:'+ '\033[0m' ' %.2f%%' % (auc*100))
print('\033[1m'+"Kappa Score:"+ '\033[0m',metrics.cohen_kappa_score(Y_test, Y_prediction_rf))
print('\033[1m'+"Mean Absolute Error:"+ '\033[0m',metrics.mean_absolute_error(Y_test, Y_prediction_rf))
print('\033[1m'+"Mean Squared Error:"+ '\033[0m',metrics.mean_squared_error(Y_test, Y_prediction_rf))
print('\033[1m'+"Root Mean Squared Error:"+ '\033[0m',np.sqrt(metrics.mean_squared_error(Y_test, Y_prediction_rf)))

fpr, tpr, thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title("ROC curve for Random Forest classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

gaussian_classification = GaussianNB()
gaussian_classification.fit(x_train, Y_train)

pickle.dump(gaussian_classification,open('gaussian_classification.pickle', 'wb'))
with open('gaussian_classification.pickle', 'rb') as g:
    gf = pickle.load(g)

print(open('gaussian_classification.pickle', 'rb').read()[:40])

#pickled gaussian classification prediction
gauf = gf.predict(x_test)
print(gauf)
probability = gf.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", gauf[:10])

print(metrics.confusion_matrix(Y_test,gauf))
print(metrics.classification_report(Y_test,gauf))

Y_prediction_gnbc = gaussian_classification.predict(x_test)
confusion = metrics.confusion_matrix(Y_test, Y_prediction_gnbc)
print("Test", Y_test[:10])
print("Prediction", Y_prediction_gnbc[:10])
print(metrics.confusion_matrix(Y_test,Y_prediction_gnbc))
print(metrics.classification_report(Y_test, Y_prediction_gnbc))

from sklearn import metrics
print("****For Gaussian Naive Bayes Classifier****")
print()
print('\033[1m'+"Accuracy Score:"+ '\033[0m',metrics.accuracy_score(Y_test, Y_prediction_gnbc)) 
print('\033[1m'+"F1 Score:"+ '\033[0m',metrics.f1_score(Y_test, Y_prediction_gnbc))
print('\033[1m'+"Precision Score:"+ '\033[0m',metrics.precision_score(Y_test, Y_prediction_gnbc))
print('\033[1m'+"Recall Score:"+ '\033[0m',metrics.recall_score(Y_test, Y_prediction_gnbc))

from sklearn.metrics import roc_auc_score, roc_curve
Y_gnb_score = gaussian_classification.predict_proba(x_test)
Y_gnb_score = Y_gnb_score[:, 1]
auc = roc_auc_score(Y_test, Y_gnb_score)
print('\033[1m'+'AUC:'+ '\033[0m' ' %.2f%%' % (auc*100))
print('\033[1m'+"Kappa Score:"+ '\033[0m',metrics.cohen_kappa_score(Y_test, Y_prediction_gnbc))
print('\033[1m'+"Mean Absolute Error:"+ '\033[0m',metrics.mean_absolute_error(Y_test, Y_prediction_gnbc))
print('\033[1m'+"Mean Squared Error:"+ '\033[0m',metrics.mean_squared_error(Y_test, Y_prediction_gnbc))
print('\033[1m'+"Root Mean Squared Error:"+ '\033[0m',np.sqrt(metrics.mean_squared_error(Y_test, Y_prediction_gnbc)))

Y_gnb_score = gaussian_classification.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_gnb_score)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Gaussian Naive Bayes Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(x_train, Y_train)

pickle.dump(decision_tree,open('decision_tree.pickle', 'wb'))
with open('decision_tree.pickle', 'rb') as d:
    dt = pickle.load(d)

print(open('decision_tree.pickle', 'rb').read()[:40])

#pickled decision tree prediction
decit = dt.predict(x_test)
print(decit)
probability = dt.predict_proba(x_test)

print("Test", Y_test[:10])
print("Prediction", decit[:10])

print(metrics.confusion_matrix(Y_test,decit))
print(metrics.classification_report(Y_test,decit))

Y_prediction_dt = decision_tree.predict(x_test)
Y_prob_dt = decision_tree.predict_proba(x_test)

print("test", Y_test[:10])
print("pred", Y_prediction_dt[:10])
print()

print(metrics.confusion_matrix(Y_test,Y_prediction_dt))
print(metrics.classification_report(Y_test, Y_prediction_dt))

print("****For Decision Tree****")
print()
print('\033[1m'+"Accuracy Score:"+ '\033[0m',metrics.accuracy_score(Y_test, Y_prediction_dt)) 
print('\033[1m'+"F1 Score:"+ '\033[0m',metrics.f1_score(Y_test, Y_prediction_dt))
print('\033[1m'+"Precision Score:"+ '\033[0m',metrics.precision_score(Y_test, Y_prediction_dt))
print('\033[1m'+"Recall Score:"+ '\033[0m',metrics.recall_score(Y_test, Y_prediction_dt))

from sklearn.metrics import roc_auc_score, roc_curve
Y_prob_dt = decision_tree.predict_proba(x_test)
Y_prob_dt = Y_prob_dt[:, 1]
auc = roc_auc_score(Y_test, Y_prob_dt)
print('\033[1m'+'AUC:'+ '\033[0m' ' %.2f%%' % (auc*100))
print('\033[1m'+"Kappa Score:"+ '\033[0m',metrics.cohen_kappa_score(Y_test, Y_prediction_dt))
print('\033[1m'+"Mean Absolute Error:"+ '\033[0m',metrics.mean_absolute_error(Y_test, Y_prediction_dt))
print('\033[1m'+"Mean Squared Error:"+ '\033[0m',metrics.mean_squared_error(Y_test, Y_prediction_dt))
print('\033[1m'+"Root Mean Squared Error:"+ '\033[0m',np.sqrt(metrics.mean_squared_error(Y_test, Y_prediction_dt)))

Y_prob_dt = decision_tree.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob_dt)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()



