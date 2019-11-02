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
 
# Create your views here.
@login_required
def index(request):
    return(render(request,"ml/index.html"))
@login_required
def indexml(request):
    if request.POST:
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