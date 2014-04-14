#! /bin/bash

mkdir -p sequences/cube-rotate
rm sequences/cube-rotate/*
mkdir -p sequences/cube-translate
rm sequences/cube-translate/*
mkdir -p sequences/cube-deform
rm sequences/cube-deform/*
mkdir -p sequences/cube-deform-rotate
rm sequences/cube-deform-rotate/*
mkdir -p sequences/cube-f-on-face
rm sequences/cube-f-on-face/*
mkdir -p sequences/female-head-deform
rm sequences/female-head-deform/*
mkdir -p sequences/female-head-rotate
rm sequences/female-head-rotate/*

cd ../synth/build/

rm *.obj
cp ../../data/models/cube.obj model.obj
./synth 0 b
rm model.obj
cp *.obj ../../data/sequences/cube-rotate

rm *.obj
cp ../../data/models/cube.obj model.obj
./synth 5 b
rm model.obj
cp *.obj ../../data/sequences/cube-translate

rm *.obj
cp ../../data/models/cube.obj model.obj
./synth 4 b
rm model.obj
cp *.obj ../../data/sequences/cube-deform

rm *.obj
cp ../../data/models/cube.obj model.obj
./synth 6 b
rm model.obj
cp *.obj ../../data/sequences/cube-deform-rotate

rm *.obj
cp ../../data/models/cube.obj model.obj
./synth 3 b
rm model.obj
cp *.obj ../../data/sequences/cube-f-on-face

rm *.obj
cp ../../data/models/femalehead.obj model.obj
./synth 4 b
rm model.obj
cp *.obj ../../data/sequences/female-head-deform

rm *.obj
cp ../../data/models/femalehead.obj model.obj
./synth 0 b
rm model.obj
cp *.obj ../../data/sequences/female-head-rotate
