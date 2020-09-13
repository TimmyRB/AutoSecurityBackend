#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import recognize


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AutoSecurity.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        )
    recognize.train("media", model_save_path="trained_knn_model.clf")
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
