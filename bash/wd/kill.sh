#!/bin/bash
tmux_sessions=$(tmux list-sessions -F "#{session_name}")

if [ -z "$tmux_sessions" ]; then
  echo "No tmux sessions found."
else
  for session in $tmux_sessions; do
    tmux kill-session -t "$session"
    echo "Killed tmux session: $session"
  done
fi
echo "All tmux sessions have been killed."

program_name="gpu_maintain.py"
procs=$(ps -ef | grep "$program_name" | awk '{print $2}')

for pid in $procs; do
    kill $pid
done