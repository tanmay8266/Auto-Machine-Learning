from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse
from django.contrib.auth.models import User,auth
from django.contrib.auth.decorators import login_required
import json
from researchera.models import Research,Files
from datetime import date
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import pickle
import os
# import required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import re
from sklearn import preprocessing
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
        elif('col3' in request.POST):
            col3 = request.POST['col3']
            col4 = request.POST['col4']
            print(col3,col4,request.session['proj_id'],request.session['name'])
            data = pd.read_csv(request.session['clean_file_link'])
            if col3==col4:
                f,ax=plt.subplots(1,2,figsize=(12,6))
                expl = len(data[col3].value_counts())
                explode_list = [0]*expl
                data[col3].value_counts().plot.pie(explode=explode_list,autopct='%1.1f%%',ax=ax[0],shadow=True)
                ax[0].set_title(col3)
                ax[0].set_ylabel('')
                sns.countplot(col3,data=data,ax=ax[1])
                ax[1].set_title(col3)
                plt.savefig("media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col3+col4))
                rdata = {"url":"/media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col3+col4)}
                return(JsonResponse(rdata))
            else:
                f,ax=plt.subplots(1,2,figsize=(12,6))
                data[[col3,col4]].groupby([col3]).mean().plot.bar(ax=ax[0])
                ax[0].set_title(col3+' vs '+col4)
                sns.countplot(col3,hue=col4,data=data,ax=ax[1])
                ax[1].set_title(col3+" vs "+col4)
                plt.savefig("media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col3+col4))
                rdata = {"url":"/media/temp/%s.png"%(request.session['name']+str(request.session['proj_id'])+col3+col4)}
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
            if(action=="clean"):
                target = request.POST["target"]
                request.session["target"] = target
                filename = request.POST["filename"]
                data = pd.read_csv('media/media/'+filename)
                print(filename,"read was successful")
                le = preprocessing.LabelEncoder()
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
                            data[i[0]].fillna(maxm_occ[0],inplace=True)
                print("Analysing Unique Values in a columns as columns with more unique values are likely not to contribute")
                print(data.isnull().sum())
                for i in data.columns:
                        print("Analysing Column",i)
                        no_uniq = data[i].nunique()
                        print(no_uniq)
                        print(data.shape[0])
                        print(no_uniq/data.shape[0])
                        if(no_uniq/data.shape[0]>0.1):
                            print("Dropped",str(i))
                            data = data.drop(str(i),axis=1)
                        else:
                            print("no uniq before",no_uniq)
                            # print(data[i])
                            data[i] = le.fit_transform(data[i])
                            if (no_uniq>(data.shape[0]*(0.09))):
                                data[i] = pd.cut(x=data[i],bins=math.ceil(math.sqrt(no_uniq)),labels=False)
                                # info_bins = pd.cut(x=data[i],bins=math.ceil(math.sqrt(no_uniq)))
                                print("nuniq after",data[i].nunique())
                if(data[target].dtypes == "object"):
                    data[target] = le.fit_transform(data[target]) 
                X=pd.get_dummies(data.drop(target,1))
                y=data[target]
                data = pd.get_dummies(data)
                print(data.head())
                data.isnull().sum()
                reg = LassoCV()
                reg.fit(X, y)
                print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
                print("Best score using built-in LassoCV: %f" %reg.score(X,y))
                coef = pd.Series(reg.coef_, index = X.columns)                                
                print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
                cor = data.corr()
                #Correlation with output variable
                cor_target = abs(cor[target])
                #Selecting highly correlated features
                relevant_features = cor_target[cor_target>0.2]
                print(relevant_features)
                data.to_csv(r'media/media/temp_cleaned_'+filename)
                file = Files(remarks="Auto Cleaned Data",link='media/temp_cleaned_'+filename,date=date.today(),name="clean_"+filename,specifications="",pro_id=request.session['proj_id'],re_id=request.user.id)
                file.save()
                print("id : ",request.user.id,request.session['proj_id'])
                request.session["filename"] = 'temp_cleaned_'+filename
                columns = list(data.columns)
                request.session["clean_file_link"]='media/media/temp_cleaned_'+filename
                rdata = {"result":"Successful","clean_url":'/media/media/temp_cleaned_'+filename,"columns":json.dumps(columns)}
                return(JsonResponse(rdata))
            elif(request.POST["action"]=="train"):
                target = request.session["target"]
                filelink = request.POST["clean_file_link"]
                data = pd.read_csv(filelink[1:])
                first_column = data.columns[0]
                no_uniq = data[target].nunique()
                # Delete first
                data = data.drop([first_column], axis=1)
                print("target_col",data[target])
                survived_train = data[target]
                data = data.drop(target,axis=1)
                train_data,eval_data,labels,eval_labels = train_test_split(data, survived_train, random_state = 0)
                # train_data = data.values[:600]
                # labels = survived_train[:600]
                # eval_data = data.values[600:]
                # eval_labels = survived_train[600:]
                if(no_uniq==2):
                    model = LogisticRegression(fit_intercept=True)
                    model.fit(train_data, labels)
                    eval_predictions = model.predict(eval_data)
                    print("Using Logistic Regression")
                else:
                #     from sklearn.tree import DecisionTreeClassifier 
                #     model = DecisionTreeClassifier(max_depth = 3).fit(train_data, labels) 
                #     eval_predictions = model.predict(eval_data)
                    
                    from sklearn.svm import SVC 
                    model= SVC(kernel = 'linear', C = 1).fit(train_data, labels) 
                    eval_predictions = model.predict(eval_data)
                    print("Using SVC") 
                # eval_predictions = model.predict(eval_data)
                print('Accuracy of the model on train data: {0}'.format(model.score(train_data, labels)))
                print('Accuracy of the model on eval data: {0}'.format(model.score(eval_data, eval_labels)))
                size = data.shape
                col = size[1]
                columns = list(data.columns)
                pkl_filename = "media/media/"+request.session["filename"].split(".")[0]+"_model.pkl"
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(model, file)
                rdata = {"result":"success","columns":json.dumps(columns),"col":col}
                return(JsonResponse(rdata))
            elif(request.POST['action']=="predict"):
                single_x_test = request.POST.getlist('col_input[]')
                print(single_x_test)
                
                pkl_filename = "media/media/"+request.session["filename"].split(".")[0]+"_model.pkl"
                with open(pkl_filename, 'rb') as file:
                    model = pickle.load(file)
                for i in range(len(single_x_test)):
                    single_x_test[i]=int(single_x_test[i])
                single_x_test = np.array(single_x_test)
                q = model.predict(single_x_test.reshape(1,-1))
                print(q[0]) 
                rdata = {"prediction":int(q[0])}
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