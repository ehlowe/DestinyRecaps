# Generated by Django 5.0.4 on 2024-07-13 05:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("destinyapp", "0016_alter_streamrecapdata_plot_image"),
    ]

    operations = [
        migrations.AlterField(
            model_name="streamrecapdata",
            name="chunk_annotations",
            field=models.JSONField(default=list),
        ),
    ]