from django.shortcuts import render, redirect

from .models import Questionnaire


# Create your views here.
def index(request):
    return render(request, 'study/index.html')


def start(request):
    questionnaire_list = Questionnaire.objects.all()
    context = {
        'questionnaire_list': questionnaire_list,
    }
    return render(request, 'study/start.html', context=context)


def about(request):
    return render(request, 'study/about.html')


def study(request):
    return redirect('start')


def questionnaire(request, questionnaire_id):
    return render(request, 'study/questionnaire.html', dict(questionnaire_id=questionnaire_id))
