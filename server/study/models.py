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

    def get_trials(self):
        return self.trial_set.order_by('order')


class Trial(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.CASCADE)
    trial_heading_text = models.CharField(max_length=200)
    order = models.IntegerField()

    def get_explanation(self):
        return self.explanation_set.get()

    def get_evaluation(self):
        return self.evaluation_set.get()


class Video(models.Model):
    static_path = models.CharField(max_length=1000)
    height = models.IntegerField()
    width = models.IntegerField()
    md5 = models.CharField(max_length=32)


class Explanation(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)


class Evaluation(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    num_choices = models.IntegerField()
    solution = models.IntegerField()


# User related classes
class Respondent(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)


class Response(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    respondent = models.ForeignKey(Respondent, on_delete=models.CASCADE)
    choice = models.IntegerField()
