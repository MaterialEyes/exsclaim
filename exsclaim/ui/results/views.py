from django.db.models import query
from django.shortcuts import render
from django.db.models.query import QuerySet
from django.http import HttpResponse
from django.views.generic import ListView
from django.db.models import Q
import os.path
import sys
from . import models

class SearchResultViews(ListView):
    model = models.Subfigure
    template_name = "exsclaim/search_results.html"


    def get_http_parameters(self):
        return self.request.GET.dict()

    def get_context_data(self):
        query = self.get_http_parameters()
        # get raw uri without page param for pagination. The page param
        # will be appended by next page button in display
        raw_uri = self.request.get_raw_uri()
        try:
            path, params = raw_uri.split("?")
            params_list = params.split("&")
            params_list = [param for param in params_list if not param.startswith("page=")]
            pageless_uri = path + "?" + "&".join(params_list)
        except:
            pageless_uri = raw_uri + "?"
        ## Set up pagination
        images_per_page = int(query.get("pagination", 50))
        paginator, page, subfigures_list, is_paginated = (
            self.paginate_queryset(self.get_queryset(), images_per_page)
        )
        columns = int(query.get("columns", 4))
        if query.get("view", "list") == "grid":
            list_of_subfigures_lists = []
            for i in range(columns):
                sub_list = subfigures_list[i::columns]
                list_of_subfigures_lists.append(sub_list)
            results_template = "exsclaim/grid_results.html"
        else:
            list_of_subfigures_lists = subfigures_list
            results_template = "exsclaim/list_results.html"
        data = {
            'list_of_subfigures_list': list_of_subfigures_lists,
            'paginator': paginator,
            'page_obj': page,
            'is_paginated': is_paginated,
            'columns': columns,
            'results_template': results_template,
            'uri': pageless_uri,
            'query': query,
        }

        return data

    def get_queryset(self):
        queryset = models.Subfigure.objects.all()
        query = self.get_http_parameters()


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
            scale_confidence = float(query.get("scale_confidence", 30)) / 100
            scale_bar_ids = models.ScaleBarLabel.objects.filter(label_confidence__gt=scale_confidence).values_list('scale_bar_id', flat=True)
            subfigure_ids = models.ScaleBar.objects.filter(scale_bar_id__in=scale_bar_ids).values_list('subfigure_id', flat=True)
            queryset = queryset.filter(subfigure_id__in=subfigure_ids)
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