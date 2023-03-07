count=10
num=0
for i in $(seq $count); do
    echo $((i+$num))
    python libs/synthetic_colored_mnist/generate_fraction_env_MULTI_SPURIOUS.py -n $((i+$num))
done