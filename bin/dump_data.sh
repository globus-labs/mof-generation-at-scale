#! /bin/bash

mongod --dbpath ./db --port 13454 --wiredTigerCacheSizeGB 2 &> export.log &
mongopid=$!
sleep 15
mongoexport --collection mofs --db mofa --port 13454 --type json --out mofs.json
kill $mongopid
gzip --verbose --best mofs.json
