#!/bin/bash


#################### HELPER FUNCTIONS ###########################
# Function: merge_default_args
#   Takes three arguments:
#     1) name of the defaults array (flat: flag value flag value …)
#     2) name of the user_args array (same format)
#     3) name of the output array
# Produces an array containing only the default flags,
# but with values overridden by any matching user_args.
# Usage:
#   merge_default_args default_args user_args final_args
merge_default_args() {
  local def_name="$1" user_name="$2" out_name="$3"
  # namerefs into the arrays whose names were passed in
  local -n defs="$def_name"
  local -n users="$user_name"
  local -n out="$out_name"

  # Build an ordered list of the default flags + a map of default → value
  declare -a keys=()
  declare -A map=()
  for ((i=0; i<${#defs[@]}; i+=2)); do
    key="${defs[i]}"
    val="${defs[i+1]}"
    keys+=( "$key" )
    map["$key"]="$val"
  done

  # Override defaults with any user-supplied values
  for ((i=0; i<${#users[@]}; i+=2)); do
    key="${users[i]}"
    val="${users[i+1]}"
    if [[ -n "${map[$key]+_}" ]]; then
      map["$key"]="$val"
    fi
  done

  # Flatten back into the out array, preserving original order
  out=()
  for key in "${keys[@]}"; do
    out+=( "$key" "${map[$key]}" )
  done
}
#################################################################


echo ""
echo "Training is being started in a tmux session..."

# Ensure TMUX_SCRIPT_DIRECTORY is set, default to the current directory if not.
: ${TMUX_SCRIPT_DIRECTORY:=$(pwd)}
echo "TMUX_SCRIPT_DIRECTORY is set to: ${TMUX_SCRIPT_DIRECTORY}"

###############################################
# Default settings for library and workflow.
###############################################
# Valid workflow options:
#   "isaaclab"  - use the library training script (for rsl_rl or skrl)
#   "eureka"    - use the eureka training script
#   "ray"       - use the ray tuner and run a separate ray_server session
WORKFLOW="isaaclab"

# The library (only rsl_rl and skrl are allowed; default is rsl_rl)
LIBRARY="rsl_rl"

# Default arguments for the isaaclab workflow (for rsl_rl and skrl)
default_args=(--task "SBTC-Unscrew-Franka-OSC-RNN-v0" --headless --livestream 0  --seed 10 --curriculum_multiplier 0.6666667)

# Array to hold additional (user-supplied) arguments.
user_args=()

###############################################
# Parse command-line arguments.
###############################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Library selection
        --library)
            shift
            LIBRARY="$1"
            shift
            ;;
        --rsl_rl)
            LIBRARY="rsl_rl"
            shift
            ;;
        --skrl)
            LIBRARY="skrl"
            shift
            ;;
        # Workflow selection
        --workflow)
            shift
            WORKFLOW="$1"
            shift
            ;;
        --eureka)
            WORKFLOW="eureka"
            shift
            ;;
        --ray)
            WORKFLOW="ray"
            shift
            ;;
        # Future options can be added here.
        *)
            # Collect any other parameter into user_args.
            user_args+=("$1")
            shift
            ;;
    esac
done

###############################################
# Library sanity check and assignment of LIBRARY_SCRIPT.
###############################################
if [ "$LIBRARY" = "rsl_rl" ]; then
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    RAY_CFG_CLASS="UnscrewFrankaRslRlJobCfg"
    RAY_METRIC="Episode_Reward/screw_engaged"
elif [ "$LIBRARY" = "skrl" ]; then
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/skrl/train.py"
    RAY_CFG_CLASS="UnscrewFrankaSkrlJobCfg"
    RAY_METRIC="Info_/_Episode_Reward/screw_engaged"
else
    echo "WARNING: Library '$LIBRARY' is not recognized. Defaulting to 'rsl_rl'."
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    RAY_CFG_CLASS="UnscrewFrankaRslRlJobCfg"
    RAY_METRIC="Episode_Reward/screw_engaged"
    LIBRARY="rsl_rl"
fi

###############################################
# Workflow sanity check and TRAINING_SCRIPT assignment.
###############################################
if [ "$WORKFLOW" = "isaaclab" ]; then
    TRAINING_SCRIPT="${LIBRARY_SCRIPT}"
    final_args=("${default_args[@]}" "${user_args[@]}")
    SESSION_NAME="isaaclab_training"
elif [ "$WORKFLOW" = "eureka" ]; then
    TRAINING_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/_isaaclab_eureka/scripts/train.py"
    # For eureka, do not prepend default arguments.
    final_args=("${user_args[@]}")
    SESSION_NAME="eureka_training"
elif [ "$WORKFLOW" = "ray" ]; then
    TRAINING_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/ray/tuner.py"
    # Define default arguments for ray tuner.
    ray_default_args=(--cfg_file "${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/ray/sbtc_ray/unscrew_franka_cfg.py" \
                      --cfg_class "${RAY_CFG_CLASS}" \
                      --run_mode "local" \
                      --workflow "${LIBRARY_SCRIPT}" \
                      --num_workers_per_node 1 \
                      --repeat_run_count 1 \
                      --num_samples 24 \
                      --metric "${RAY_METRIC}" \
                      --mode "max" \
                      --process_response_timeout 120.0 \
                      --max_lines_to_search_experiment_logs 1000 \
                      --max_log_extraction_errors 2 \
                      )
    merge_default_args ray_default_args user_args final_args  #  merge defaults with any overrides in user_args
    SESSION_NAME="ray_training"
else
    echo "WARNING: Workflow '$WORKFLOW' is not recognized. Defaulting to 'isaaclab'."
    WORKFLOW="isaaclab"
    TRAINING_SCRIPT="${LIBRARY_SCRIPT}"
    final_args=("${default_args[@]}" "${user_args[@]}")
    SESSION_NAME="isaaclab_training"
fi

###############################################
# Print the command settings for verification.
###############################################
echo "Selected library: ${LIBRARY}"
echo "Using workflow: ${WORKFLOW}"
echo "Using training script: ${TRAINING_SCRIPT}"
echo "With arguments: ${final_args[@]}"
sleep 3  # Optional: allow the user to see the output before proceeding.


###############################################
# TMUX session and logging configuration.
###############################################

LOG_DIR="${TMUX_SCRIPT_DIRECTORY}/tmux"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="${LOG_DIR}/${WORKFLOW}/${TIMESTAMP}_${LIBRARY}.log"

# Ensure the log directory exists (including the workflow subdirectory).
mkdir -p "${LOG_DIR}/${WORKFLOW}"

TRAIN_CMD="cd ${TMUX_SCRIPT_DIRECTORY} && ${TMUX_SCRIPT_DIRECTORY}/_isaac_sim/python.sh ${TRAINING_SCRIPT} ${final_args[@]} | tee -a ${LOG_FILE}"

###############################################
# Ensure we are running outside any tmux session.
###############################################

if [ -n "$TMUX" ]; then
    echo "Detected tmux session. Detaching from current tmux and re-running script outside tmux..."
    tmux detach-client
    exec env -u TMUX "$0" "$@"
fi

###############################################
# If workflow is ray, handle the ray_server session first.
###############################################
if [ "$WORKFLOW" = "ray" ]; then
    RAY_SERVER_SESSION="ray_server"
    RAY_SERVER_CMD='echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p'
    if ! tmux has-session -t "${RAY_SERVER_SESSION}" 2>/dev/null; then
        echo "Creating new tmux session '${RAY_SERVER_SESSION}' for Ray server..."
        tmux new-session -d -s "${RAY_SERVER_SESSION}" -n ray_server "${RAY_SERVER_CMD}; bash"
    else
        echo "Ray server session '${RAY_SERVER_SESSION}' already exists. Sending command..."
        tmux send-keys -t "${RAY_SERVER_SESSION}:ray_server" "$RAY_SERVER_CMD" C-m
    fi
    echo "Ray server session '${RAY_SERVER_SESSION}' is being started in 20 seconds..."
    sleep 20
    echo "Ray server session '${RAY_SERVER_SESSION}' is ready."
    sleep 1
fi

###############################################
# --- TMUX Session Handling for isaaclab/eureka/ray_training ---
###############################################

# If not in a tmux session, create or send the command to the target session.
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Target tmux session '$SESSION_NAME' does not exist. Creating new session..."
    tmux new-session -d -s "${SESSION_NAME}" -n training "${TRAIN_CMD}; bash"
else
    echo "Target tmux session '$SESSION_NAME' already exists."
    tmux send-keys -t "${SESSION_NAME}:training" "$TRAIN_CMD" C-m
fi

# Finally, attach to the target session.
tmux attach-session -t "${SESSION_NAME}"
