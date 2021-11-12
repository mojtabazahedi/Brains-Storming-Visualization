from django.shortcuts import render, redirect
from django.shortcuts import HttpResponse
from .forms import InputForm
from .models import Input

# Create your views here.


def home(request):
    return render(request, 'end.html')


def user_input(request):
    context = {"form": InputForm}
    return render(request, 'input_form.html', context)


def add_input(request):
    form = InputForm(request.POST)
    if form.is_valid():
        #input_item = form.save(commite=False)
        #input_item.save()
        inputs = Input(author=form.cleaned_data['author'], input=form.cleaned_data['input'])

        inputs.save()

    return render(request, 'end.html')


