from django.urls import path
from .views import SearchResultViews

urlpatterns = [
    path('', SearchResultViews.as_view(), name="search_results"),
]