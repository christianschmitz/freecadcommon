#!/bin/bash
FILE=FreeCADCommon.py
TARGET=`readlink -e $FILE`
DEST=`readlink -e $HOME/python`/${FILE}

if [ ! -h $DEST ]
then
  echo "Linking $DEST to $TARGET"
  ln -s $TARGET $DEST
else
  echo "Link already exists, not changing anything"
fi

