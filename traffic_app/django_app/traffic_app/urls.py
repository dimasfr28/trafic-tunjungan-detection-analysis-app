from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analysis/', views.analysis, name='analysis'),
    path('prediction/', views.prediction, name='prediction'),
    path('api/video/<str:hour>/', views.get_video_for_hour, name='get_video'),
    path('api/metrics/', views.get_metrics, name='get_metrics'),
    path('api/chart-data/', views.get_chart_data, name='chart_data'),
]
