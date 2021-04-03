from django.urls import path
from .views import query_view

urlpatterns = [
    path('', query_view, name="query_page"),
]