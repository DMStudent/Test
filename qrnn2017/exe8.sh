exe=run.py
nohup python ${exe}  --job_name=ps --task_index=0 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=1 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=2 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=3 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=4 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=5 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=6 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
nohup python ${exe}  --job_name=ps --task_index=7 1>results/stdout_ps.txt 2>results/stderr_ps.txt &
for i in 0 2 3 4 5 6 7
do
    gpuid=$[$i % 8]
    echo "No.$i worker submit on gpu $gpuid"
    nohup python ${exe}  --job_name=worker --task_index=${i} --deviceid=${gpuid} 1>results/stdout_wk${i}.txt 2>results/stderr_wk${i}.txt &
done
python ${exe}  --job_name=worker --task_index=1 --deviceid=1