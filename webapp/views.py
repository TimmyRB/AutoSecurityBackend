from django.shortcuts import render
from rest_framework import viewsets, views
from .models import Student, Camera
from .serializers import StudentSerializer, CameraSerializer
from recognize import Recognition

# Create your views here.

def FaceFound(name):
    print(name)
    # Do code for websockets here

class StudentView(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer
    Recognition.train("media", "trained_knn_model.clf")

class CameraView(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer

class Run(views.APIView):
    def put(self, request, pk, format=None):
        Recognition.run_recognition(Camera.objects.first().address, FaceFound)

    @classmethod
    def get_extra_actions(cls):
        return []
    
