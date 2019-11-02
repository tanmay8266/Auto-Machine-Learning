from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib.auth.models import User,auth
from .models import Research,Files
from django.http import JsonResponse
from datetime import date
from django.core.files.storage import FileSystemStorage
from django.urls import reverse
from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Create your views here.
from django.contrib.auth.decorators import login_required

 
@login_required
def index(request):
    if(request.POST):
        name = request.POST['nor']
        ini_pitch = request.POST['ini_pitch']
        iden = request.user.id
        if name=="":
            return render(request,"home/index.html")
        else:
            re = Research(name=name,ini_pitch=ini_pitch,re_id=iden)
            re.save()
            return JsonResponse({'status':'ok'})
    else:
        return render(request,"home/index.html",{'name':request.user.first_name})
@login_required
def projects(request):
    if('proj_id' in request.POST):
        request.session['proj_id'] = request.POST['proj_id']
        return(redirect("project/"))
    else:
        if('proj_id' in request.session):
            del request.session['proj_id']  
        projects = Research.objects.filter(re_id=request.user.id)
        return render(request, "projects/index.html",{'projects':projects})

@login_required
def project(request):
    if(request.POST): 
        if('document' in request.FILES):
            project = Research.objects.filter(id=request.POST['proj_id'])
            files = Files.objects.filter(pro_id=request.POST['proj_id'])
            uploaded_file = request.FILES['document']
            file = Files(remarks=request.POST['remarks'],link=request.FILES['document'],date=date.today(),name=uploaded_file.name.split(".")[0],specifications=uploaded_file.name.split(".")[1],pro_id=request.POST['proj_id'],re_id=request.user.id)
            file.save()
            request.session['proj_id'] = request.POST['proj_id']
            request.session['uploaded'] = 'Yes'
            return(redirect("/home/projects/project/",{'project':project,'files':files}))
        elif('proj_id' in request.POST):
            project = Research.objects.filter(id=request.POST['proj_id'])
            files = Files.objects.filter(pro_id=request.POST['proj_id'])
            return(render(request,"projects/project/index.html",{'project':project,'files':files}))
    else:
        if('proj_id' in request.session):
            project = Research.objects.filter(id=request.session['proj_id'])
            files = Files.objects.filter(pro_id=request.session['proj_id'])
            if('uploaded' in request.session):
                del request.session['uploaded']
                return(render(request,"projects/project/index.html",{'project':project,'files':files,'uploaded':'1'}))
            return(render(request,"projects/project/index.html",{'project':project,'files':files}))
        else:
            return(redirect("/home/projects/"))
        # fs = FileSystemStorage()
        # fs.save(uploaded_file.name,uploaded_file)
@login_required
def relevant(request):
    proj_id = request.session['proj_id']
    project = Research.objects.filter(id=request.session['proj_id'])
    stop_words = set(stopwords.words('english'))
    stop_words.add('the')
    pitch = project.values('ini_pitch')[0]['ini_pitch']
    print(pitch)
    word_tokens = word_tokenize(pitch) 
    filtered = [w for w in word_tokens if not w in stop_words] 
    filtered = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered.append(w)
    string = ""
    for word in filtered:
        try:
            val = int(word)
            string = string[0:len(string)-1] + word
        except ValueError:
            string = string + word + "%20%"
    query = string[0:len(string)-4]
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q='+query+'&btnG='
    content = requests.get(url).text
    page = BeautifulSoup(content, 'lxml')
    results = []
    i=0
    for entry in page.find_all("div",class_="gs_ri"):
        i+=1
        abst = entry.find('div',attrs={'class':'gs_rs'})
        auth = entry.find('div',attrs={'class':'gs_a'})
        divfl = entry.find('div',attrs={'class':'gs_fl'})
        try:
            results.append({"title": entry.a.text, "url": entry.a['href'],"abstract": abst.text,'author':auth.text,'cited':divfl.find_all('a')[2].text})
        except:
            pass
        if(i==5):
            break
    return(render(request,'projects/project/relevant/index.html',{'results':results}))
@login_required
def logout(request):
    auth.logout(request)
    return redirect("/login/")