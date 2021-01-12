from django.shortcuts import render, redirect


# Create your views here.
def index(request):
    return render(request, 'study/index.html')


def start(request):
    return render(request, 'study/start.html')


def about(request):
    return render(request, 'study/about.html')


def study(request):
    return redirect('start')


def questionnaire(request, questionnaire_id):
    return render(request, 'study/questionnaire.html', dict(questionnaire_id=questionnaire_id))
