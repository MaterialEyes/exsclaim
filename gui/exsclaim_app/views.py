#from django.core.paginator import QuerySetPaginator
from django.db.models.query import QuerySet
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView, ListView
from django.db.models import Q
from . import models

# Create your views here.



class SearchResultViews(ListView):
    context_object_name = "subfigures_list"
    model = models.Subfigure
    template_name = "exsclaim/search_results.html"
    paginate_by = 50
    #queryset = models.Subfigure.objects.all()
    def get_queryset(self):
        queryset = models.Subfigure.objects.all()
        query = self.request.GET.dict()

        # Filter based on classification first (reducing queryset size looking
        # at exact on a 2 char field, seems more efficient)
        if "classification[]" in query:
            classifications = self.request.GET.getlist("classification[]")
            q_expression = Q()
            for classification in classifications:
                q_expression |= Q(classification__exact=classification)
            queryset = queryset.filter(q_expression)
        if "open" in query:
            license = True if query["open"] else False
            queryset = queryset.filter(figure__article__open=license)
        if "min_scale" in query and query["min_scale"]:
            min_scale = query["min_scale"]
            queryset = queryset.filter(nm_width__gte=min_scale)
        if "max_scale" in query and query["max_scale"]:
            max_scale = query["max_scale"]
            queryset = queryset.filter(nm_width__lt=max_scale)
        if (query.get("max_scale", "") or query.get("min_scale", "")):
            scale_confidence = query.get("scale_confidence", 0.3)
            # queryset = queryset.filter()



        if "keyword" in query:
            keyword = query["keyword"]
            if "keyword_location[]" in query:
                keyword_locations = self.request.GET.getlist("keyword_location[]")
            else:
                keyword_locations = ["full_caption"]
            q_expression = Q()
            if "caption" in keyword_locations:
                q_expression |= Q(caption__icontains=keyword)
            if "full_caption" in keyword_locations:
                q_expression |= Q(figure__caption__icontains=keyword)
            if "title" in keyword_locations:
                q_expression |= Q(figure__article__title__icontains=keyword)
            queryset = queryset.filter(q_expression)

        return queryset





class QueryView():
    pass