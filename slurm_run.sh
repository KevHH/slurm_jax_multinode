#!/bin/bash
#SBATCH --job-name=slurm_test

# helper functions
wait_for() {
    counter=$1
    shift 1
    until [[ $counter < 0 ]] || ("$@" &> /dev/null); do
        echo waiting for "$@"
        sleep 1
        counter=$(( counter - 1 ))
    done
    if [[ $counter < 0 ]]; then
        return 1
    fi
}

find_coordinator() {
    if [ -f "$log_dir/coord.ip" ]; then 
        # found file
        return 1;
    else
        # have not found file
        return 0;
    fi
}

# print current node status
IP=$(hostname -I)
echo "directory for logging: $LOGDIR"
echo "current IP: $IP"
echo "current node name: $SLURMD_NODENAME"
echo "memory: $MEM"
echo "resources: $NUM_DEVICES $DEVICE_TYPE(s)"
echo "job id: $SLURM_JOB_ID"
echo "process id: $SLURM_ARRAY_TASK_ID" 
echo "total number of processes: $NUM_JOBS"
echo "timeout: $TIMEOUT"

if [[ $SLURM_ARRAY_TASK_ID = 0 ]]; then 
    # assign coordinator node
    echo "coordinator: true"
    echo "coordination port: $PORT"
    echo "$IP" > "$LOGDIR/coord.ip"
    export COORD_IP=$IP
else 
    # retrieve coordinator node
    echo "coordinator: false"
    wait_for $TIMEOUT find_coordinator
    if [[ $? = 0 ]]; then
       export COORD_IP=$(cat "$LOGDIR/coord.ip")
        echo "coordinator ip: $COORD_IP"
    else 
        echo "coordinator not found. exiting."
        exit 1
    fi
fi

echo "===================================="

# REPLACE YOUR_ENV with your desired conda environment
# REPLACE slurm_distributed_test.py with your python script
source /PATH/TO/YOUR/CONDA/bin/activate YOUR_ENV
python slurm_distributed_test.py
