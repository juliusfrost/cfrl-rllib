from django.urls import path

from . import views

app_name = 'study'  # namespacing url names
urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start, name='start'),
    path('about/', views.about, name='about'),
    path('study/', views.study, name='study'),
    path('study/<int:questionnaire_id>/', views.questionnaire, name='questionnaire'),
    path('submit/<int:questionnaire_id>/', views.submit, name='submit'),
]
