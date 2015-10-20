#!/bin/bash

if [[ -f tasks.txt ]]; then
  rm tasks.txt
fi

ls -1 *_list.txt | sed -e 's/\..*$//' > tasks.txt
