from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import re
from . import views
urlpatterns = [
    path('',views.index,name = "index"),
    path('projects/',views.projects,name = "index"),
    path('projects/project/',views.project,name  = "project"),
    path('logout/',views.logout,name = "index"),
    path('projects/logout/',views.logout,name = "index"),
    path('projects/project/logout/',views.logout,name  = "index"),
]

