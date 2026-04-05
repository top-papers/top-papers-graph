#!/bin/sh

set -xe

cd backend

if [ ! -d venv ]; then
  python3 -m venv venv
  venv/bin/pip3 install -r requirements.txt
fi

venv/bin/python3 app.py
