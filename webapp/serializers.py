from rest_framework import serializers

from . import views
from .models import Student, Camera


class StudentSerializer(serializers.ModelSerializer):
    # def save(self):
    #     views.StudentView(self.context['request'])
    #     recognize.break_run()
    #     recognize.train("media", model_save_path="trained_knn_model.clf")
    #     recognize.run_recognition(0, faceFound)

    class Meta:
        model = Student
        fields = ('id', 'studentId', 'firstName', 'lastName', 'imageRef')


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ('id', 'address', 'port')
