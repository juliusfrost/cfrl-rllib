from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, 'study/index.html')


def start(request):
    return render(request, 'study/start.html')


def about(request):
    return render(request, 'study/about.html')


def study(request, template_name):
    return render(request, template_name)
