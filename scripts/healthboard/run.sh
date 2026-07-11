#!/bin/sh
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later


set -xe

cd backend

if [ ! -d venv ]; then
  python3 -m venv venv
  venv/bin/pip3 install -r requirements.txt
fi

venv/bin/python3 app.py
