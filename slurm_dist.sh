#!/bin/bash
set -e

# default variables
run_name=""
file=""
mem="2G"
type="gpu"
N="2"
num_devices="1"
port="1234"
log_dir=""
timetout="300"

# function to display helper info
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help        Display this help message"
 echo " -f, --file        Specify slurm job file to sbatch. Required." 
 echo " --log             Specify the folder for logging. Any existing \"coord.ip\" file in this folder will be REPLACED. Required" 
 echo " -m, --mem         Specify memory per slurm node. Default to \"2G\""
 echo " -t, --type        \"gpu\" or \"cpu\". Default to \"gpu\"" 
 echo " -N, --num_nodes   Specify the number of nodes, required to be >=2. Default to \"2\". "
 echo " --num_devices     Specify the number of devices per node. Default to \"1\". "
 echo " -p, --port        Specify the port number for communication. Default to \"1234\""
 echo " --timeout         Specify the timeout for communication in seconds. Default to \"300\""
 echo ""
 echo "Variables made available to slurm job file:"
 echo "  - \$LOGDIR                 stores --log argument."
 echo "  - \$PORT                   stores --port argument"
 echo "  - \$NUM_JOBS               stores --num_nodes argument."
 echo "  - \$SLURM_ARRAY_TASK_ID    stores --timeout argument."
 echo "  - \$SLURM_ARRAY_TASK_ID    integer to differentiate the tasks."
}

# functions to handle options
extract_argument() {
    # input: option name, $@
    # output: option value
    if [[ "$2" == *=* && -n ${1#*=} ]]; then 
        echo "${2#*=}"
        echo 1
    elif [[ -n "$3" ]]; then 
        echo "$3"
        echo 2
    else
        echo "$1 not specified, exiting."
        echo 0
    fi
}

# read options
while [ $# -gt 0 ]; do
    case $1 in
        -h | --help)
            usage
            break
            ;;
        -m | --mem*)
            readarray -t output <<<"$(extract_argument "memory" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                mem=${output[0]}; shift ${output[1]}
            fi
            ;;
        -t | --type*)
            readarray -t output <<<"$(extract_argument "type" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                type=${output[0]}; shift ${output[1]}
            fi
            ;;
        -N | --num_nodes*)
            readarray -t output <<<"$(extract_argument "number of nodes" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                N=${output[0]}; shift ${output[1]}
            fi
            ;;
        -p | --port*)
            readarray -t output <<<"$(extract_argument "port" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                port=${output[0]}; shift ${output[1]}
            fi
            ;;
        -f | --file*)
            readarray -t output <<<"$(extract_argument "file" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                file=${output[0]}; shift ${output[1]}
            fi
            ;;
        --log*)
            readarray -t output <<<"$(extract_argument "directory for logging" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                log_dir=${output[0]}; shift ${output[1]}
            fi
            ;;
        --num_devices*)
            readarray -t output <<<"$(extract_argument "number of devices" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                num_devices=${output[0]}; shift ${output[1]}
            fi
            ;;
        --timeout*)
            readarray -t output <<<"$(extract_argument "timeout" $@)"
            if [[ ${output[1]} == 0 ]]; then 
                usage; exit; break
            else 
                timeout=${output[0]}; shift ${output[1]}
            fi
            ;;
        *)
            echo "Invalid option: $1, exiting."; usage; exit 1
            break
            ;;
    esac
done

if [[ $type = "gpu" ]]; then 
    resource="-p gpu --gres=gpu:$num_devices"
else 
    resource=""
fi

if [[ $file = "" ]]; then 
    echo "job file not specified, exiting."
    usage 
    exit 1
fi

if [[ $log_dir = "" ]]; then 
    echo "logging directory not specified, exiting."
    usage 
    exit 1
fi

if [[ $((N)) < 2 ]]; then 
    echo "$N node(s) requested but at least two nodes required, exiting."
    usage 
    exit 1
fi

# remove existing coord.ip
if [ -f "$log_dir/coord.ip" ]; then 
    rm "$log_dir/coord.ip" 
fi
wait

# sbatch
echo "sbatch $file on $N nodes at port $port, with $num_devices $type(s) and $mem memory per node..."
mkdir -p $log_dir
sbatch --array="0-$((N-1))" -N $N $resource --mem=$mem -o $log_dir/slurm-%A_%a.out \
        --export=NUM_JOBS=$N,PORT=$port,LOGDIR=$log_dir,MEM=$mem,NUM_DEVICES=$num_devices,DEVICE_TYPE=$type,TIMEOUT=$timeout \
        ./$file