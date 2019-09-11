from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
# Create your views here.
def register(request):
    if request.method == "POST":
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        password = request.POST['password']
        password_repeat = request.POST['password-repeat']
        organization = request.POST['organization']

        user = User.objects.create_user(password=password,username=email,first_name=first_name,last_name=last_name)
        user.save()
        print("User Created")
        return redirect("/login/")
    else:
        return render(request,"register/index.html")
def login(request):
    next = ""
    if request.GET:
        next = request.GET['next']
    if(request.method =='POST'):
        username = request.POST['email']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            if next == "":
                return redirect("/home/")
            else:
                return redirect(next)
        else:
            messages.info(request,'invalid credentials')
            return render(request,"login/index.html",{"alert":"invalid credentials"})
    else:
        if request.user.is_authenticated:
            return redirect("/home/")
        else:
            return render(request,"login/index.html")
