from django.contrib.auth.models import User
from django.db import models


class Questionnaire(models.Model):
    name = models.CharField(max_length=1000)
    num_trials = models.IntegerField()
    intro_text = models.CharField(max_length=1000)
    title_text = models.CharField(max_length=200)
    explanation_study_text = models.CharField(max_length=1000)
    eval_study_text = models.CharField(max_length=1000)
    explain_heading_text = models.CharField(max_length=200)
    eval_heading_text = models.CharField(max_length=200)
    question_text = models.CharField(max_length=1000)


class Trial(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.CASCADE)
    trial_heading_text = models.CharField(max_length=200)
    order = models.IntegerField()


class Explanation(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    static_path = models.CharField(max_length=1000)


class Evaluation(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    static_path = models.CharField(max_length=1000)
    num_choices = models.IntegerField()
    solution = models.IntegerField()


# User related classes
class Respondents(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)


class Response(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    choice = models.IntegerField()
