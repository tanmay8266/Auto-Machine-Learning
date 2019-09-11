from django.contrib import admin

# Register your models here.
from .models import Research,Files

admin.site.register(Research)
admin.site.register(Files)