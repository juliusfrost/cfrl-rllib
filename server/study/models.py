from django.db import models


class Questionnaire(models.Model):
    name = models.CharField(max_length=1000)


class User(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.CASCADE)


class Question(models.Model):
    question_text = models.CharField(max_length=255)  # optional
    questionnaire = models.ForeignKey(Questionnaire, models.CASCADE)
    trial = models.IntegerField()


class Choice(models.Model):
    choice_text = models.CharField(max_length=255)  # optional
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    votes = models.IntegerField(default=0)


class Response(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    choice = models.ForeignKey(Choice, on_delete=models.CASCADE)


class Solution(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice = models.ForeignKey(Choice, on_delete=models.CASCADE)
