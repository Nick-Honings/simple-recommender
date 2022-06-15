#!/usr/bin/env bash
set -e

virtualenv virtualenv
source virtualenv/bin/activate
pip install -r requirements.txt
deactivate