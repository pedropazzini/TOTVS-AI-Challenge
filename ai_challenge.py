import json
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import squareform,pdist,cdist
from scipy.spatial.distance import mahalanobis
import numpy as np
from sklearn import ensemble
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV,train_test_split,KFold
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import xgboost as xgb

def parse_date_info(dic):
    '''
    Parse the date info from the dictionary contained in dic

    Returns:
    --------
    tuple_: with 6 elements
        
        year
        date
        month
        day
        hour
        minute
        weekday
        week
    '''

    str_date = dic['ide']['dhEmi']['$date'].replace('T',' ').replace('.000Z','')
    date = datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')    

    day = date.day
    week = 0
    if day <=9:
        week = 1
    elif day <= 16:
        week = 2
    elif day <= 23:
        week = 3
    elif day <= 30:
        week = 4


    return date.year, date.month, date.day, date.hour,date.minute,date.weekday(),week

def load_data():
    '''
    Loads the data in json format contained in the 'sample.txt' file.

    Returns:
    --------
    full_data: list

    List of dictionaries containing the data of the file 'sample.txt'
    '''
    with open('sample.txt','r') as f:
        full_data = json.load(f)

    return full_data

def build_dataframe(full_data):
    '''
    Build a pandas data frame from the data contained in the 'sample.txt' file.

    Returns:
    --------

    tuple: (pd.DataFrame A, pd.DataFrame B)
        A: Pandas dataframe containing general data of a transaction
        B: Pandas dataframe containing data of items of each transaction
    '''
    Matrix_data = []
    Matrix_items = []
    pk = 0
    for data in full_data:

        pk += 1
        year, month, day, hour, minute, weekday, week = parse_date_info(data)
        natOp = data['ide']['natOp']
        infoAdc = data['infAdic']['infCpl']
        valorTotal = data['complemento']['valorTotal']
        cnpj = data['emit']['cnpj']

        Matrix_data.append([pk,year,month,day,hour,minute,weekday,week,natOp,infoAdc,cnpj,valorTotal])

        for item in data['dets']:

            prod_dic = item['prod']

            total = prod_dic['indTot']
            qCom = prod_dic['qCom']
            uCom = prod_dic['uCom']
            vProd = prod_dic['vProd']
            vUnCom = prod_dic['vUnCom']
            xProd = prod_dic['xProd']

            Matrix_items.append([pk,total,qCom,uCom,vProd,vUnCom,xProd])

    df_items = pd.DataFrame(Matrix_items,columns=['pk','total','qCom','uCom','vProd','vUnCom','xProd'])
    df_data = pd.DataFrame(Matrix_data,columns=['pk','year','month','day','hour','minute','weekday','week','natOp','infoAdc','cnpj','valorTotal'])

    return df_data,df_items

def build_X_y_from_dataframes(df_data,df_items):

    M = []
    product_cols = B['xProd'].unique()
    for ix,row in df_data.iterrows():

        slice_df_items = df_items[df_items['pk']==row.pk]
        row_ = []
        for product in product_cols:
            row_.append(np.sum(slice_df_items[slice_df_items.xProd==product].qCom))

        M.append(row_)

    unique_tables = df_data['infoAdc'].unique()
    tables = np.array([np.where(i==unique_tables)[0] for i in df_data['infoAdc'].values]).reshape(-1,1)

    y = df_data['valorTotal'].values.reshape(-1,1)
    X = np.hstack((df_data[['day','hour','minute','weekday','week']].as_matrix(),tables,M))

    df = pd.DataFrame(X,columns = np.append(['day','hour','minute','weekday','week','tables'],product_cols))

    return df,y

def remove_outliers_by_mahalanobis(X):
    '''
    Removes the outliers using the mahalanobis distance from the mean to each instance. By simplicity, the outliers will be removed supposing a guassian distribution.
    TODO: Apply KernerlDensity Estimation to remove outliers

    Parameters:
    -----------
    X: ndarray (n_samples,n_features)
        Matrix data

    Returns:
    --------
    mask: ndarray (n_samples,)
        Boolean mask representing the data that is not an outlier.

    '''

    mu_ = np.mean(X,axis=0) #finds the mean point

    C = np.cov(X)
    C_ = C+np.eye(C.shape[0])*(np.random.uniform(C.shape[0],C.shape[1])/1e6) # Add noise to the diagonal of the covariance matrix to avoid singular matrix

    VI = np.linalg.inv(C_)

    d_mahalanobis = cdist([mu_],X,metric='mahalanobis',VI=VI)[0]
    d_mahalanobis_ = d_mahalanobis[~np.isnan(d_mahalanobis)]

    limit = np.mean(d_mahalanobis_) + 2.0*np.std(d_mahalanobis_)
    mask_ = d_mahalanobis_<limit

    mask = np.ones((X.shape[0],),dtype=bool)
    mask[~np.isnan(d_mahalanobis)] = mask_

    return mask

def predict_last_week(df,y,clf,kde,mask):
    '''
    Predicts the sales forecast of the next week.

    Parameters:
    -----------

    Returns:
    --------
    tuple: (min_forecast,forecast,max_forecast)
        Tuple formed by the interval (min,max) of the forecast with 95% confidence centered in the forecasted value (excluding the error of the classifier)
    '''

    r = df.groupby(['week']).size()

    df = df[df.columns[mask]]

    n_tables_week_2 = r[2]
    n_tables_week_3 = r[3]
    limit_inferior = min(n_tables_week_2,n_tables_week_3)
    limit_superior = max(n_tables_week_2,n_tables_week_3)
    #predicts and generates 3000 times the samples generated from a random number of costumers

    predictions = np.array([np.sum(clf.predict(kde.sample(np.random.randint(limit_inferior,limit_superior)))) for i in range(3000)])
    mu_ = np.mean(predictions)
    std_ = np.std(predictions)
    forecast = (mu_-2*std_,mu_,mu_+2*std_)

    return forecast

def fit_kde(X):
    '''
    Fits a density function to the data in X

    Parameters:
    -----------
    X: ndarray (n_samples,n_features)
        Matrix data to be fitted

    Returns:
    --------
    kde: KernelDensity
        The kernel density estimator that best fits the data in 'X'
    '''
    grid_cv = GridSearchCV(KernelDensity(),{'bandwidth': np.linspace(0.01, 1.0, 300),'kernel':['gaussian','tophat']},cv=20,verbose=3,n_jobs=-1)
    grid_cv.fit(X)

    return grid_cv.best_estimator_

def get_top_n_important_features(X,y,n,method,cols=None):
    '''
    Select the top important features
    '''

    if method == 'ensemble_tree':
        return get_top_n_important_features_by_ensemble_tree(X,y,n,cols)
    elif method == 'mutual_info':
        sk = SelectKBest(mutual_info_regression,k=n)
        sk.fit(X,y)
        return sk.get_support()
    elif method == 'f_regression':
        sk = SelectKBest(f_regression,k=n)
        sk.fit(X,y)
        return sk.get_support()
    else:
        raise NotImplementedError()

def get_top_n_important_features_by_ensemble_tree(X,y,n,cols):
    '''
    Select top n features second a tree ensemble classifier.
    '''
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X,y)
    feature_importance = clf.feature_importances_
    sorted_importance = np.argsort(feature_importance)[::-1]

    mask = np.zeros((X.shape[1],),dtype=bool)
    
    for i in sorted_importance[:n]:
        mask[i] = True

    return mask

def fit_regressor(df,y,clf,n_features=5,feauture_method='ensemble_tree',exclude_first_week=True,**params_clf):
    '''
    Fits a regressor classifier for the data

    Parameters:
    -----------


    Returns:
    --------
    A vector containing the results of the input data
    '''

    if exclude_first_week:
        df = df[df.week>1]
        y = y[(df.week>1).values]

    cols = df.columns

    # Splits tha data
    X = df.as_matrix()
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    # Select top n features

    # Evaluate the strategy in test data
    mask_features = get_top_n_important_features(x_train,y_train,n_features,feauture_method,df.columns)
    x_train_ = x_train[:,mask_features]
    x_test_ = x_test[:,mask_features]

    mask_non_outliers = remove_outliers_by_mahalanobis(x_train_)
    x_train__ = x_train_[mask_non_outliers,:]
    y_train__ = y_train[mask_non_outliers]

    clf.fit(x_train__,y_train__.reshape(-1,))
    mse = mean_squared_error(y_test,clf.predict(x_test_))

    return mse,mask_features,clf,X

def get_classifier(clf_name):

    if clf_name == 'svm':
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid={"C": [1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)},scoring='neg_mean_squared_error',verbose=3)
        return svr
    elif clf_name == 'xgb':
        xgb_model = xgb.XGBRegressor()
        clf = GridSearchCV(xgb_model,
                                   {'max_depth': [2,4,6],
                                       'n_estimators': [50,100,200]}, scoring='neg_mean_squared_error',verbose=3)
        return clf

def main():

    full_data = load_data()
    data_general,data_items = build_dataframe(full_data)

    df,y = build_X_y_from_dataframes(data_general,data_items)
    X = df.as_matrix()

    clf = get_classifier('svm')
    #clf = get_classifier('xgb')
    mse_test,mask_features,clf_fitted,x_train = fit_regressor(df,y,clf,n_features=9,feauture_method='ensemble_tree')

    final_clf = clf_fitted.best_estimator_
    final_clf.fit(X,y)

    kde = fit_kde(X)

    forecast = predict_last_week(df,y,final_clf,kde,mask_features)

    return forecast,mse_test,mask_features,clf_fitted,df.columns

def caller_main():

    forecast,mse_test,mask_features,clf_fitted,cols = main()
    print('Test error: ' + str(mse_test))
    print('CV error: ' + str(abs(clf_fitted.best_score_)))
    print('Most relevant features: ' + ', '.join(cols[mask_features]))
    print('MIN FORECAST SALES FOR THE NEXT WEEK: ' + str(forecast[0]))
    print('MAX FORECAST SALES FOR THE NEXT WEEK: ' + str(forecast[2]))
    print('FORECAST SALES FOR THE NEXT WEEK: ' + str(forecast[1]))

if __name__ == "__main__":
    caller_main()

