#!/bin/bash


LAYER1=('1000' '400'  '200' '100') #layer size
LAYER2=('20' '100' '200' '600') #layer size

#List of datasets:
#path to dataset NSLKDD


for k in "${LAYER1[@]}"
do
	for f in "${LAYER2[@]}" #layers couple: try 1 layers combinations
	do
		python stl_3.py "$k" "$f"
	done
done



