from django.http import Http404
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, AnonymousUser

from .models import Questionnaire, Response, Respondents


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
    try:
        q = Questionnaire.objects.get(pk=questionnaire_id)
    except Questionnaire.DoesNotExist:
        raise Http404("Questionnaire does not exist")
    context = {'questionnaire': q}
    return render(request, 'study/questionnaire.html', context=context)


def submit(request, questionnaire_id):
    q = Questionnaire.objects.get(pk=questionnaire_id)
    if isinstance(request.user, AnonymousUser):
        user = None
    else:
        user = request.user
    respondent = Respondents(questionnaire=q, user=user)
    context = {
        'user': request.user,
        'choices': [],
        'score': 0,
    }
    for trial in q.get_trials():
        explanation = trial.get_explanation()
        evaluation = trial.get_evaluation()
        name = f'trial_{trial.order}'
        choice = int(request.POST[name])
        context['choices'].append(choice)
        if choice == evaluation.solution:
            context['score'] += 1
        response = Response(trial=trial, user=user, choice=choice)

    return render(request, 'study/submit.html', context)
