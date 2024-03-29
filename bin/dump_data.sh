#! /bin/bash

mongod --dbpath ./db --port 13454 &> export.log &
mongopid=$!
mongoexport --collection mofs --db mofa --port 13454 --type json --out mofs.json
gzip --verbose --best mofs.json
kill $mongopid
