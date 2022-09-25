###################
# LIBRARIES
###################

import joblib
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.ticker import AutoMinorLocator
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################
# FUNCTIONS
###################

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
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

breast_cancer = pd.read_csv("_breast_cancer/Breast_Cancer.csv")
df = breast_cancer.copy()

check_df(df)
df.info()

df.rename(columns={"Reginol Node Positive":"Regional Node Positive",
                   "T Stage ":"T Stage"}, inplace=True)
df.isnull().any()


##T0: No evidence of primary tumor. T1 (includes T1a, T1b, and T1c):
# Tumor is 2 cm (3/4 of an inch) or less across. T2:
# Tumor is more than 2 cm but not more than 5 cm (2 inches) across.
# T3: Tumor is more than 5 cm across.

##N2: Cancer has spread to 4 to 9 lymph nodes under the arm,
# or cancer has enlarged the internal mammary lymph nodes.
# N2a: Cancer has spread to 4 to 9 lymph nodes under the arm,
# with at least one area of cancer spread larger than 2 mm.

num_cols = [col for col in df.columns if df[col].dtypes != "object"]
cat_cols = [col for col in df.columns if df[col].dtypes == "object"]

df.head()
df.Status.value_counts()
le = LabelEncoder()

df["Status"] = le.fit_transform(df["Status"])

for col in num_cols:
    target_summary_with_num(df, "Status", col)

for col in cat_cols:
    print("Variable: " + str(col))
    target_summary_with_cat(df, "Status", col)

df["Grade"].value_counts()

df.drop(df.loc[(df["Grade"] == " anaplastic; Grade IV") & (df["differentiate"] == "Undifferentiated"), :].index,
        inplace=True)

df["Node_Ex_to_Pos"] = df["Regional Node Examined"] / df["Regional Node Positive"]

sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.show()

df.columns

for col in cat_cols:
    cat_summary(df, col, False)

stage_cols = [col for col in df.columns if "Stage" in col]

for col in stage_cols:
    fig = plt.figure(figsize=(6,7))
    g = sns.countplot(x=df[col], hue=df["Status"], dodge=True,
                    palette="viridis")
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    show_values(g)
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    fig.savefig(str(col), dpi=300)

for col in stage_cols:
    temp_df = df.groupby(col)["Status"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
    fig = plt.figure(figsize=(6,7))
    g = sns.barplot(x=temp_df[col], y=temp_df["percent"], hue=temp_df["Status"], dodge=False,
                    palette="viridis", ci=None)
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    show_values(g)
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    fig.savefig(str(col), dpi=300)
## Train_test_split to avoid data leakage when it comes to transformation

train, test = train_test_split(df, test_size=.2, stratify=df.Status, random_state=26)

# Count plot
for col in stage_cols:
    fig = plt.figure(figsize=(6,7))
    g = sns.countplot(x=train[col], hue=train["Status"], dodge=True,
                    palette="viridis")
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    show_values(g)
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    fig.savefig(str(col), dpi=300)

# Stacked and standardized bar plot (dodge=False/True)
for col in stage_cols:
    temp_df = train.groupby(col)["Status"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
    fig = plt.figure(figsize=(6,7))
    g = sns.barplot(x=temp_df[col], y=temp_df["percent"], hue=temp_df["Status"], dodge=True,
                    palette="viridis", ci=None)
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    show_values(g)
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    fig.savefig(str(col), dpi=300)

# Distributions
for col in num_cols:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=train[col], kde=False, color="purple", hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Variable: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(2))
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show(block=True)

## Right skew: Tumor Size, Regional Node Examined, Regional Node Positive, Node_Ex_to_Pos
## Left skew: Age, Survival Months

df.columns

# train["Sqrt_Regional_Node_Positive"] = np.sqrt(train["Regional Node Positive"])
# train["Inv_Regional_Node_Positive"] = 1 / train["Regional Node Positive"]

fig = plt.figure(figsize=(8, 6))
g = sns.distplot(x=np.log1p(train["Node_Ex_to_Pos"]), kde=False, color="purple", hist_kws=dict(edgecolor="black", linewidth=2))
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)


plt.figure(figsize=(9, 6))
g = sns.distplot(x=np.log1p(train["Tumor Size"]), kde=True, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Log Tumor Size")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)

train["Log_Node_Ex_to_Pos"] = np.log1p(train["Node_Ex_to_Pos"])
train["Log_Regional_Node_Positive"] = np.log1p(train["Regional Node Positive"])
train["Log_Tumor_Size"] = np.log1p(train["Tumor Size"])

num_cols = [col for col in train.columns if train[col].dtypes != "object"]
cat_cols = [col for col in train.columns if train[col].dtypes == "object"]

for col in num_cols:
    fig = plt.figure(figsize=(8,6))
    g = sns.distplot(x=train[col], kde=False, color="purple", hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Variable: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(2))
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show(block=True)

train.drop(["Node_Ex_to_Pos", "Regional Node Positive", "Tumor Size"], axis=1, inplace=True)
num_cols = [col for col in train.columns if train[col].dtypes != "object"]
cat_cols = [col for col in train.columns if train[col].dtypes == "object"]
train = one_hot_encoder(train, cat_cols, drop_first=True)


test["Log_Node_Ex_to_Pos"] = np.log1p(test["Node_Ex_to_Pos"])
test["Log_Regional_Node_Positive"] = np.log1p(test["Regional Node Positive"])
test["Log_Tumor_Size"] = np.log1p(test["Tumor Size"])
test.drop(["Node_Ex_to_Pos", "Regional Node Positive", "Tumor Size"], axis=1, inplace=True)
num_cols = [col for col in test.columns if test[col].dtypes != "object"]
cat_cols = [col for col in test.columns if test[col].dtypes == "object"]
test = one_hot_encoder(test, cat_cols, drop_first=True)

## MODELS
X_train = train.drop("Status", axis=1)
y_train = train["Status"]
X_test = test.drop("Status", axis=1)
y_test = test["Status"]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")

cv_results = cross_validate(lr_model,
                            X_train, y_train,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results["test_f1"].mean()

## To take a different approach towards imbalance of the data in respect to churn
## we may change the class_weight accordingly and search through best values for F1 score

lr = LogisticRegression(max_iter=1000, random_state=26)

#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr,
                          param_grid= param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X_train, y_train)

#Ploting the score for different values of weight
fig = plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
plt.show(block=True)
fig.savefig("score_for_different_values", dpi=300)

weigh_data.sort_values(by="score", ascending=False).iloc[0, :]

lr = LogisticRegression(max_iter=1000, random_state=26, class_weight={0: .264, 1: .736}).fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(random_state=26)

gridsearch = GridSearchCV(estimator= lgbm,
                          param_grid= param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X_train, y_train)

#Ploting the score for different values of weight
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
plt.show(block=True)

weigh_data.sort_values(by="score", ascending=False).iloc[0, :]

lgbm = LGBMClassifier(random_state=26, class_weight={0: .284, 1: .716}).fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

lgbm_params = {"learning_rate": [0.01, 0.1, 0.2],
               "n_estimators": [300, 500, 800, 1000],
               "colsample_bytree": [0.3, 0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

lgbm_final = lgbm.set_params(**lgbm_best_grid.best_params_, random_state=26).fit(X_train, y_train)

y_pred = lgbm_final.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=26)

gridsearch = GridSearchCV(estimator= rf,
                          param_grid= param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X_train, y_train)

#Ploting the score for different values of weight
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
plt.show(block=True)

weigh_data.sort_values(by="score", ascending=False).iloc[0, :]

rf = RandomForestClassifier(random_state=26, class_weight={0: .806, 1: .194}).fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

rf.get_params()

rf_params = {"max_depth": [3, 5, 8, 12, None],
             "max_features": [3, 5, 7, 10, "sqrt"],
             "min_samples_split": [8, 15, 20, 25],
             "n_estimators": [300, 500, 800, 1000]}

rf_best_grid = GridSearchCV(rf,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_train, y_train)

rf_final = RandomForestClassifier(**rf_best_grid.best_params_, random_state=26).fit(X_train, y_train)

y_pred = rf_final.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

def plot_importance(model, features, num=len(X_test), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm, X_test, 10, True)