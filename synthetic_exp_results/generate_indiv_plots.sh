count=13
num=1
for i in $(seq $count); 
do
    python generate_performance_plot.py -d $((i-$num))
done