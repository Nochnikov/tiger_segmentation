from AI import back
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from myapp.serlizers import PhotoUploadSerializer
import base64

class UpLoadPhotoView(APIView):
    serializer_class = PhotoUploadSerializer

    def post(self, request, *args, **kwargs):

        photo = self.request.FILES.get('photo')

        binary_segmented = back.receive_segmented_data(photo)
        binary_segmented = base64.b64encode(binary_segmented)

        response_message = {"status": "photo was successfully uploaded",
                            "segmented": binary_segmented}

        return Response(response_message, status=status.HTTP_200_OK)


