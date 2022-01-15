#!/bin/bash

cd $(dirname $0)
docker run --rm -it -v $PWD:$PWD -w $PWD terf/dcombat distributedCombat_comparison.py
rm *.pickle
