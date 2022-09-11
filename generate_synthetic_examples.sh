count=10
num=2
for i in $(seq $count); do
    echo $((i+$num))
    python libs/synthetic_colored_mnist/generate_multiple_fraction_env.py -n $((i+$num))
done