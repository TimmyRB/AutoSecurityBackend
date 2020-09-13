from django.db import models

# Create your models here.

def get_upload_path(instance, filename):
    return "%s/%s" % (instance.studentId, filename)

class Student(models.Model):
    studentId = models.CharField(max_length=32)
    firstName = models.CharField(max_length=32)
    lastName = models.CharField(max_length=32)
    imageRef = models.FileField(upload_to=get_upload_path)

    def __str__(self):
        return self.firstName + " " + self.lastName


class Camera(models.Model):
    address = models.GenericIPAddressField(protocol='IPv4')
    port = models.PositiveSmallIntegerField()

    def __str__(self):
        return self.address + ':' + str(self.port)
