#!/bin/bash

MONGOD=`which mongod`
PORT=28000
DBPATH=~/scratch/mongodb/
LOGPATH=~/scratch/mongodb/mongodb.log

$MONGOD --fork --logpath $LOGPATH --port $PORT --dbpath $DBPATH
