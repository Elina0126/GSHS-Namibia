import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from MissForestExtra import MissForestExtra
import sklearn.cluster
import sys
import scipy
import Gap_statistics
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


def result(df,labels): #show the result of clustering 
    a = ['11','12','13','14','15','16','17','18','Female','Male'] #Q1 and Q2 need to be treated specially
    b = []
    for i in df.columns:
        a.append(i)
        b.append(i)
    a.remove('Q1')
    a.remove('Q2')
    b.remove('Q1')
    b.remove('Q2')
    df_result = pd.DataFrame(0,columns = a,index = np.unique(labels)) 
    #create a data frame with rows number equal to the number of clusters and columns number equal to the original data frame plus age number and gender number
    for i in range(len(df)):
        for j in b:
            df_result.loc[labels[i],j]+=df.loc[i,j]
        #Count the number of each age     
        if int(float(df.loc[i,'Q1'])) == 11:
            df_result.loc[labels[i],'11']+=1
        elif int(float(df.loc[i,'Q1'])) == 12:
            df_result.loc[labels[i],'12']+=1
        elif int(float(df.loc[i,'Q1'])) == 13:
            df_result.loc[labels[i],'13']+=1    
        elif int(float(df.loc[i,'Q1'])) == 14:
            df_result.loc[labels[i],'14']+=1
        elif int(float(df.loc[i,'Q1'])) == 15:
            df_result.loc[labels[i],'15']+=1
        elif int(float(df.loc[i,'Q1'])) == 16:
            df_result.loc[labels[i],'16']+=1
        elif int(float(df.loc[i,'Q1'])) == 17:
            df_result.loc[labels[i],'17']+=1
        else:
            df_result.loc[labels[i],'18']+=1 

        if int(float(df.loc[i,'Q2'])) == 2:
            df_result.loc[labels[i],'Female']+=1
        else:
            df_result.loc[labels[i],'Male']+=1

    return df_result
    
    
def elbow_kmode(df,int1,int2,name):
    cost = []
    K = range(int1,int2)
    init_type = ['Cao', 'Huang', 'random']
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = 'Cao', n_init = 5, verbose=1)
        kmode.fit_predict(df)
        cost.append(kmode.cost_)

    fig, axs = plt.subplots(1)
    fig.suptitle('Elbow Method For Optimal k')
    axs.plot(K, cost[0:5],'bx-')
    fig.savefig("elbow_"+str(name)+".jpg")

def elbow_kmeans(df,name):
    y = []
    z = []
    for i in range(2,11):
        kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=0).fit(df)
        y.append(kmeans.inertia_)
        z.append(silhouette_score(df,kmeans.labels_))
    x = np.linspace(2,10,9)
  #  fig, ax = plt.subplots(2)
    
    plt.figure(figsize=(16,5))
    
    plt.subplot(1,2,1) 
    plt.title("Elbow methods")
    plt.plot(x,y)
    
    plt.subplot(1,2,2) 
    plt.title("Silhouette score")
    plt.plot(x,z)
    
    plt.show
    plt.savefig("Find optimal number_"+str(name)+"_gf.png")

def gf_(df):
    col_list_dietary_behaviours = ['QN6','QN7','QN8','QN9','QN10']
    col_list_hygiene = ['QN11','QN12','QN13','QN14']
    col_list_injury = ['QN15','QN16','QN17','QN18','QN19','QN20','QN21']
    col_list_mental_health = ['QN22','QN23','QN24','QN25','QN26','QN27']
    col_list_tobacco_use = ['QN28','QN29','QN30','QN31','QN32','QN33']
    col_list_alcohol_use = ['QN34','QN35','QN36','QN37','QN38','QN39']
    col_list_drug_use = ['QN40','QN41','QN42','QN43']
    col_list_sexual_behaviours = ['QN44','QN45','QN46','QN47','QN48']
    col_list_physical_activity = ['QN49','QN50','QN51','QN52']
    col_list_protective_factors = ['QN53','QN54','QN55','QN56','QN57','QN58']
    df['dietary'] = df[col_list_dietary_behaviours].mean(axis=1)
    df['hygiene'] = df[col_list_hygiene].mean(axis=1)
    df['injury'] = df[col_list_injury].mean(axis=1)
    df['mental_health'] = df[col_list_mental_health].mean(axis=1)
    df['tobacco_use'] = df[col_list_tobacco_use].mean(axis=1)
    df['alcohol_use'] = df[col_list_alcohol_use].mean(axis=1)
    df['drug_use'] = df[col_list_drug_use].mean(axis=1)
    df['sexual_behaviours'] = df[col_list_sexual_behaviours].mean(axis=1)
    df['physical_activity'] = df[col_list_physical_activity].mean(axis=1)
    df['protective_factors'] = df[col_list_protective_factors].mean(axis=1)
    a=[]
    for i in range(2,55):
        a.append(i)
    df.drop(df.columns[a], axis=1, inplace=True)
    return df

def kprototypes_elbow(df,name):
    df['Q1'] = df['Q1'].apply(str)
    df['Q2'] = df['Q2'].apply(str)
    df.select_dtypes('object').nunique()
    catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
    dfMatrix = df.to_numpy()
    cost = []
    x = np.linspace(1,6,6)
    for cluster in range(1,7):
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
    plt.figure(figsize=(8,5))
    plt.title("Elbow methods")
    plt.plot(x,cost)
    plt.show
    plt.savefig("Find optimal number for kp_"+str(name)+"_gf.png")
    
def kprototypes_(df,int1):
#int1 is the number of clusters getting from elbow-method
    df['Q1'] = df['Q1'].apply(str)
    df['Q2'] = df['Q2'].apply(str)
    df.select_dtypes('object').nunique()
    catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
    dfMatrix = df.to_numpy()
    kprototype = KPrototypes(n_jobs = -1, n_clusters = int1, init = 'Huang', random_state = 0)
    kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
    results_ = result(df,kprototype.labels_)
    return results_
