#!/usr/bin/env bash
cd /root/fairseq 
. $HOME/.profile
#killall luajit 
source /root/.bashrc 


trainjob=$(ps -Al |grep luajit)

echo "$trainjob"

if ["$trainjob" -eq ""];
then
		echo "not has job,starting"
		./train-zh.sh >nohup.out 2>&1 &
else
		echo "has job"
fi

#
