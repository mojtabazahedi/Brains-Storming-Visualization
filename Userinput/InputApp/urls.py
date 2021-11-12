

from django.urls import path
from .views import user_input, add_input

urlpatterns = [
    path('', user_input, name='Input'),
    path('addInput', add_input, name='addInput')
]
