from rest_framework import serializers
from .models import Student, Camera

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = ('id', 'studentId', 'firstName', 'lastName', 'imageRef')

class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ('id', 'address', 'port')