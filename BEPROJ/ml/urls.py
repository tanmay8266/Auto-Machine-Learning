from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import re
from . import views
urlpatterns = [
    path('',views.index,name = "index"),
    path('ml',views.indexml,name = "ml")
]

