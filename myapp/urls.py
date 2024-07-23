from django.urls import path
from myapp.views import UpLoadPhotoView

urlpatterns = [
    path('', UpLoadPhotoView.as_view(), name='upload_photo'),
]

