# Generated by Django 2.2.4 on 2019-09-07 11:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('researchera', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='research',
            name='re_id',
            field=models.IntegerField(),
            preserve_default=False,
        ),
    ]