from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib.auth.models import User,auth
# Create your views here.
def index(request):
    return(render(request,"ml/index.html"))