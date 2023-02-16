from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('add', views.add, name='add'),
    path('handleFileUpload', views.handleFileUpload, name='handleFileUpload'),
    path('train-test', views.showTrainTest, name='showTrainTest')
]