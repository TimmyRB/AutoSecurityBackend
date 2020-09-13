from django.shortcuts import render
from rest_framework import viewsets, views
from rest_framework.decorators import action
from .models import Student, Camera
from .serializers import StudentSerializer, CameraSerializer
import recognize


# Create your views here.

def faceFound(name):
    print(name)
    # Do code for websockets here


class StudentView(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer

    @action(detail=True, methods=['post'])
    def runAI(self, request, format=None):
        print("I'm going to kill Steve")
        recognize.break_run()
        recognize.train("media", model_save_path="trained_knn_model.clf")
        recognize.run_recognition(0, faceFound)


class CameraView(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
