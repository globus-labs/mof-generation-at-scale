#! /bin/bash


# Compress cubes and log files
bzip2 -9 --verbose $1/all_ligands.csv
ls "$/*-results.json" | xargs -P 4 -n 1 bzip2 -9 --verbose
find $1 -name "*.cube" | xargs -P 4 -n 1 bzip2 -9 --verbose
find $1/logs -name "*.log" | xargs -P 4 -n 1 bzip2 -9 --verbose
find $1/logs -name "*.csv" | xargs -P 4 -n 1 bzip2 -9 --verbose
