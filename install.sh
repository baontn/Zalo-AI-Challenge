#!/usr/bin/env bash

python3 -m venv zac_env
source zac_env/bin/activate
pip install -r requirements.txt
python3 main.py run
