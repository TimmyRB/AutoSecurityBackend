from django.db import models

# Create your models here.
class Students(models.Model):
    firstName = models.CharField(max_length=12)
    lastName = models.CharField(max_length=12)
    imageRef = models.CharField(max_length=128)

    def __str__(self):
        return self.firstName + " " + self.lastName