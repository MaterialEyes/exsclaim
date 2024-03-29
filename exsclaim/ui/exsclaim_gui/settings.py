"""
Django settings for exsclaim_gui project.

Generated by 'django-admin startproject' using Django 3.1.6.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

from configparser import ConfigParser
from pathlib import Path
import os.path


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'fakekey'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'exsclaim.ui.query',
    'exsclaim.ui.results',
    'exsclaim.ui.home',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'csp.middleware.CSPMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'exsclaim.ui.exsclaim_gui.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'results' / 'templates',
            BASE_DIR / 'query' / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'exsclaim.ui.exsclaim_gui.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

# just using a random string as password. this is unsafe for anything
# other than locally running the ui
configuration_file = BASE_DIR.parent / "utilities" / "database.ini"
parser = ConfigParser()
parser.read(configuration_file)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'exsclaim',
        'USER': parser["exsclaim"]["user"],
        'PASSWORD': parser["exsclaim"]["password"], 
        'HOST': 'localhost',
        'PORT': '',
    }
}


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static",
    BASE_DIR.parent.parent / "extracted"
]
# Add base_dirs (base directories for saving extractions) to static file dirs
if os.path.isfile(BASE_DIR.parent / "results_dirs"):
    with open(BASE_DIR.parent / "results_dirs", "r") as f:
        results_dirs = f.readlines()
    for base_dir in results_dirs:
        if base_dir != "":
            STATICFILES_DIRS.append(base_dir.strip())

# For django >3.2
DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'

# Content Security Policy

CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'", )
CSP_SCRIPT_SRC_ELEM = ("'self'",)
CSP_STYLE_SRC = ("'self'", )
CSP_STYLE_SRC_ELEM = ("'self'",)
CSP_STYLE_SRC_ATTR = ("'self'",)


# Celery
CELERY_BROKER_URL = 'amqp://localhost'