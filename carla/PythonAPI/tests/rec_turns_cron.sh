#!/bin/bash
PATH=/usr/local/cuda-11.0/bin:/home/yujiro/.pyenv/shims:/home/yujiro/.pyenv/bin:/home/yujiro/gems/bin:/home/yujiro/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
source /home/yujiro/git/2022-tfm-enrique-shinohara/carla/venv36/bin/activate
python /home/yujiro/git/2022-tfm-enrique-shinohara/carla/PythonAPI/tests/pilotnet_recarla_simply.py -r rgb
