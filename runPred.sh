#!/bin/bash
for s in test_sensors/*.csv;
do
	b=${s##*/}
	sensor=${b%-*}
	#for tres in 720 360;
	for tres in 180 60 30 5;
	do
		for lookback in 10;
#48;
		do
			for lr in 0.00001 0.000001;
			#for lr in 0.000001;
			do
				python etsf-pred.py -r $lookback -l $lr -S $sensor -t $tres
			done
		done
	done
done



#	for i in tp_*.py;
#	do
#		echo $i $sensor
#		python3 $i -S $sensor;

#for sensor in 01A02T010 01A03T006 01A04T008 01A05T007 01A06T013 01A07T003 01A08T005 01A09T015 01A10T012 01A11T009 01A12T011 01A13T013;
#do
#	for i in tp_*.py;
#	do
#		echo $i
#		#python3 $i -i lin -S $sensor;
#	done
#done
