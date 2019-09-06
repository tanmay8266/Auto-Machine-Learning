from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
# Create your views here.
from django.contrib.auth.decorators import login_required
 
@login_required
def index(request):
    return render(request,"home/index.html")
 
@login_required
def projects(request):
    return render(request, "projects/index.html")

@login_required
def project(request):
    return(render(request,"projects/project/index.html"))
@login_required
def logout(request):
    auth.logout(request)
    return redirect("/login/")