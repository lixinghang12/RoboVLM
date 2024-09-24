
program_name="python3 eval/calvin/evalu"
procs=$(ps -ef | grep "$program_name" | awk '{print $2}')

for pid in $procs; do
    kill $pid
done
