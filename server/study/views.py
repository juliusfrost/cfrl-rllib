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
    context = {
        'questionnaire': q,
        'trials': []
    }
    for trial in q.get_trials():
        explanation = trial.get_explanation()
        evaluation = trial.get_evaluation()
        explain_video = explanation.video
        eval_video = evaluation.video
        explain_aspect_ratio = explain_video.height / explain_video.width * 100
        eval_aspect_ratio = eval_video.height / eval_video.width * 100
        trial_context = {
            'trial': trial,
            'explanation': explanation,
            'evaluation': evaluation,
            'explain_video': explain_video,
            'eval_video': eval_video,
            'explain_aspect_ratio': explain_aspect_ratio,
            'eval_aspect_ratio': eval_aspect_ratio,
            'explain_style': f'--aspect-ratio: {explain_aspect_ratio}%;',
            'eval_style': f'--aspect-ratio: {eval_aspect_ratio}%;',
        }
        context['trials'].append(trial_context)
    return render(request, 'study/questionnaire.html', context=context)


def submit(request, questionnaire_id):
    q = Questionnaire.objects.get(pk=questionnaire_id)
    if isinstance(request.user, AnonymousUser):
        user = None
    else:
        user = request.user
    context = {
        'questionnaire': q,
        'user': request.user,
        'choices': [],
        'score': 0,
    }
    if (user is not None) and Respondents.objects.filter(user=user).exists():
        return render(request, 'study/submitted.html', context)
    respondent = Respondents(questionnaire=q, user=user)
    respondent.save()
    for trial in q.get_trials():
        explanation = trial.get_explanation()
        evaluation = trial.get_evaluation()
        name = f'trial_{trial.order}'
        choice = int(request.POST[name])
        context['choices'].append(choice)
        if choice == evaluation.solution:
            context['score'] += 1
        response = Response(trial=trial, user=user, choice=choice)
        response.save()

    return render(request, 'study/submit.html', context)
