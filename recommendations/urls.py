from django.urls import include, path

urlpatterns = [
    path('', include('recommendations.api.urls')),
]
