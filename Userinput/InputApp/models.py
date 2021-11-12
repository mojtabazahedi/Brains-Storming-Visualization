from django.db import models

# Create your models here.


class Input(models.Model):
    author = models.CharField(max_length=30)
    input = models.TextField()

    def __str__(self):
        return self.author




