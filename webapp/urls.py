from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('students', views.StudentView)
router.register('cameras', views.CameraView)

urlpatterns = [
    path('', include(router.urls))
]
