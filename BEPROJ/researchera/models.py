from django.db import models

# Create your models here.

class Research(models.Model):
    name = models.CharField( max_length=50)
    ini_pitch = models.TextField()
    re_id = models.IntegerField()
class Files(models.Model):
    name = models.CharField(max_length=50)
    date = models.DateField()
    specifications = models.CharField(max_length = 50)
    link = models.FileField(upload_to='media')
    remarks = models.TextField()
    re_id = models.IntegerField()
    pro_id = models.IntegerField()

