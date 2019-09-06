from django.urls import path, include
from . import views
urlpatterns = [
    path('',views.login,name = "index"),
    path('register/',views.register, name = "index"),
    path('home/',include("researchera.urls"))
]
