from django.db import models

# Create your models here.

class Research(models.Model):
    name = models.CharField( max_length=50)
    no_of_researchers =  models.IntegerField()
    email =  models.EmailField(max_length=254)
    ini_pitch = models.TextField()
    researcher_id = models.IntegerField()