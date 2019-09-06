from django.urls import path, include
from . import views
urlpatterns = [
    path('',views.index,name = "index"),
    path('projects/',views.projects,name = "index"),
    path('projects/project/',views.project,name  = "index"),
    path('logout/',views.logout,name = "index"),
    path('projects/logout/',views.logout,name = "index"),
    path('projects/project/logout/',views.logout,name  = "index"),
]
