1) add this file to vit_deploy/vit_classification/models:
  https://drive.google.com/file/d/1gQW7tilnqqJQc6-1aH2_fl1hWOgdo-DH/view?usp=sharing

2) Install torch, timm, django

3) to run the app:
    cd vit_deploy
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver

note: use conda 
