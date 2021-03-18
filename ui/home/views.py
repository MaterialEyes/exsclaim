from django.shortcuts import render

def home_view(request):
    template_name = "exsclaim/home.html"
    return render(request, template_name)