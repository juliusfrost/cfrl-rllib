# Setting up the user study server

## Installation

### Install requirements:
```bash
# inside cfrl-rllib/server directory
pip install -r requirements.txt
```

### Migrate database:
```bash
python manage.py migrate
```
If you are getting the error `django.db.utils.OperationalError: no such table:` then add `--run-syncdb`

### Add an admin account (optional):
```bash
python manage.py createsuperuser
```

### Run the server:
```bash
python manage.py runserver
```

## Adding additional studies

TODO: implement
