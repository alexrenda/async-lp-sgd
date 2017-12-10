#!/usr/bin/env bash

make clean && make && mv main "${1?}" && ./grid.sh "${1?}"
