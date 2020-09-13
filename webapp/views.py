import time
from datetime import datetime
import json
import threading

from bson import json_util
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets, views
from rest_framework.response import Response

import recognize
from .models import Student, Camera
from .serializers import StudentSerializer, CameraSerializer


# Create your views here.
def faceFound(names):
    print(names)
    if len(names) > 0:
        for name in names:
            if name == "Unknown":
                continue
            s = Student.objects.get(studentId=name)
            s.lastSeen = datetime.now()
            s.save()


class StudentView(views.APIView):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer

    def get(self, request):
        students = [ob.as_json() for ob in Student.objects.all()]
        return HttpResponse(json.dumps(students), content_type="application/json")

    def post(self, request):
        ai = threading.Thread(target=runapi)
        ai.start()

        s = Student()
        s.studentId = request.POST['studentId']
        s.firstName = request.POST['firstName']
        s.lastName = request.POST['lastName']
        s.imageRef = request.FILES['imageRef']

        s.save()

        return HttpResponse(json.dumps(s.as_json()), content_type="application/json")


class CameraView(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer


def runapi():
    recognize.break_run()
    recognize.train("media", model_save_path="trained_knn_model.clf")
    time.sleep(15)
    c = Camera.objects.first()
    recognize.run_recognition("http://" + str(c) + "/video", faceFound)
