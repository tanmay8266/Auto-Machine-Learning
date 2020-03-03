from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse
from django.contrib.auth.models import User,auth
from django.contrib.auth.decorators import login_required
import json
from researchera.models import Research,Files
from datetime import date
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
# import required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import math
# Create your views here.
@login_required
def index(request):
    return(render(request,"ml/index.html"))
@login_required
def indexml(request):
    if request.POST:
        print("help")
        if('col1' in request.POST):
            col1 = request.POST['col1']
            col2 = request.POST['col2']
            print(col1,col2,request.session['proj_id'],request.session['name'])
            data = pd.read_csv(os.path.join(settings.MEDIA_ROOT,"media/"+request.session['name']+".csv"))
            if col1==col2:
                f,ax=plt.subplots(1,2,figsize=(12,6))
                expl = len(data[col1].value_counts())
                explode_list = [0]*expl
                data[col1].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
                ax[0].set_title(col1)
                ax[0].set_ylabel('')
                sns.countplot(col1,data=data,ax=ax[1])
                ax[1].set_title(col1)
                plt.savefig("media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col1+col2))
                rdata = {"url":"/media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col1+col2)}
                return(JsonResponse(rdata))
            else:
                f,ax=plt.subplots(1,2,figsize=(12,6))
                data[[col1,col2]].groupby([col1]).mean().plot.bar(ax=ax[0])
                ax[0].set_title(col1+' vs '+col2)
                sns.countplot(col1,hue=col2,data=data,ax=ax[1])
                ax[1].set_title(col1+" vs "+col2)
                plt.savefig("media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col1+col2))
                rdata = {"url":"/media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col1+col2)}
                return(JsonResponse(rdata))
        elif('file' in request.FILES):
            request.session['proj_id'] = request.POST['proj_id']
            uploaded_file = request.FILES['file']
            request.session['name'] = uploaded_file.name.split(".")[0]
            data = pd.read_csv(uploaded_file)
            snapshot = data.head()
            size = data.shape
            row = size[0]
            col = size[1]
            columns = list(data.columns)
            file = Files(remarks="ML Zone File",link=request.FILES['file'],date=date.today(),name=uploaded_file.name.split(".")[0],specifications=uploaded_file.name.split(".")[1],pro_id=request.POST['proj_id'],re_id=request.user.id)
            file.save()
            rdata={"snapshot":snapshot.to_html(header="true", table_id="table"),"row":row,"col":col,"columns":json.dumps(columns)}
            return JsonResponse(rdata)
        elif('action' in request.POST):
            action = request.POST["action"]
            filename = request.POST["filename"]
            if(action=="clean"):
                data = pd.read_csv('media/media/'+filename)
                print(filename,"read was successful")
                for i,j in zip(data.columns,data.isnull().sum()/data.shape[0]>0.50):
                    if(j==True):
                        print(i," has been dropped")
                        data=data.drop(str(i),axis=1)
                alter_list = []
                for i,j in zip(data.columns,data.isnull().sum()):
                    if(j>0):
                        if(data[i].dtypes == "float64" or data[i].dtypes == "int64"):
                            data_type="Numeric"
                        elif(data[i].dtypes == "object"):
                            data_type="String"
                        else:
                            data_type=""
                        print(i,j," Data type being : ",data_type)
                        alter_list.append((i,data_type))
                for i in alter_list:
                    if i[1] == "Numeric":
                        data[i[0]] = data[i[0]].fillna(data[i[0]].mean())
                    elif i[1]=="String":
                        print("Unique values in the column",i[0],data[i[0]].unique())
                        print("Number of unique vlaues in column",i[0], data[i[0]].nunique())
                        print("Number of each unique value occurances made in column",i[0]+":\n",data[i[0]].value_counts(dropna=False))
                        toleration_limit = math.floor(data.shape[0]*0.1)
                        if data[i[0]].nunique()>toleration_limit:
                            data = data.drop(i[0],axis=1)
                        else:
                            maxm_occ = data[i[0]].value_counts(dropna=False)[:1].index.tolist()
                            print(maxm_occ[0])
                            data[i[0]].fillna(maxm_occ[0])
                print("Analysing Unique Values in a columns as columns with more unique values are likely not to contribute")
                for i in data.columns:
                        print("Analysing Column",i)
                        no_uniq = data[i].nunique()
                        print(no_uniq)
                        print(data.shape[0])
                        print(no_uniq/data.shape[0])
                        if(no_uniq/data.shape[0]>0.1):
                            data = data.drop(str(i),axis=1)
                data = pd.get_dummies(data)
                print(data.head())
                X=data.drop("Survived",1)
                y=data["Survived"]
                data.isnull().sum()
                reg = LassoCV()
                reg.fit(X, y)
                print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
                print("Best score using built-in LassoCV: %f" %reg.score(X,y))
                coef = pd.Series(reg.coef_, index = X.columns)                                
                print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
                cor = data.corr()
                #Correlation with output variable
                cor_target = abs(cor["Survived"])
                #Selecting highly correlated features
                relevant_features = cor_target[cor_target>0.2]
                
                rdata = {"result":"Successful"}
            return(JsonResponse(rdata))
        else:
            rdata = {"Content":"You've reached end of POST request!"}

    else:
        return(render(request,"ml/indexml.html"))
'''The data preprocessing algorithm'''
''' Asking the user if the columns are important'''
'''
- If user deletes the columns then proceed if he doesn't then algorithm will analyse the columns and prepare for deletion with
an advice to user
- After the deletion of columns prepare for null value analysis
- Null value analysis can be done by showing user first about null columns and ask if algorithm
can delete it
- If user permits the dataset with null values then select the columns with null values 
and if numeric data then fill with average/median
- If text then simply take which one occurs more(note this will depend the amount of nulls if nulls are 
less then only use this method)'''