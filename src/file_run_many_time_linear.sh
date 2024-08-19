for num_depth in 128 256 512 1024; do
    for step in 0.001 0.01 0.1 0.5; do
        for couling_strength in 2 3 4 5 6 7; do
            python3 run_grand_ex.py --one_block --coupling_strength $couling_strength --depth $num_depth --discritize_type norm --step_size $step --dataset Citeseer
            python3 run_grand_ex.py --one_block --coupling_strength $couling_strength --depth $num_depth --discritize_type norm --step_size $step --dataset Cora
        done
	done
done
