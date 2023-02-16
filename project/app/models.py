from django.db import models

class UploadFile(models.Model):
    csv = models.FileField(null=True)