import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-traffic-analysis-key-2024'

DEBUG = os.environ.get('DEBUG', '1') == '1'

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'traffic_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'traffic_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'traffic_project.wsgi.application'

DATABASES = {}

LANGUAGE_CODE = 'id'
TIME_ZONE = 'Asia/Jakarta'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

FASTAPI_URL = os.environ.get('FASTAPI_URL', 'http://localhost:8002')
ASSETS_DIR = BASE_DIR / 'assets'
CSV_PATH = ASSETS_DIR / 'excel' / 'vehicle_counts_fuzzy_clustered.csv'
VIDEO_DIR = ASSETS_DIR / 'video' / 'input_vidio'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
