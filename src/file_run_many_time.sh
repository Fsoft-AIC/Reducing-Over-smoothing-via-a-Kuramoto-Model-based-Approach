# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5 6 7 8 9 10; do
#     python3 run_GNN.py --run_time $run --split_rate $split 
# done
# done

# for split in 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --split_rate $split --dataset Pubmed
# done
# done
# for coup in 0.9 0.8 0.7; do
# for time in 1 4 8 16 32 64 80 100; do
# for coup in 2 3 4 5; do

# cora:
# label 1 - time 10 0.7 coup
for time in 80 100; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method euler --time $time --step_size 0.1 --no_early --coupling_strength 0.8 --dataset Citeseer
    # python3 run_GNN.py --planetoid_split --method dopri5 --kuramoto 1 --time $time --coupling_strength $coup  --dataset Citeseer --add_noise 1
    # python3 run_GNN.py --kuramoto 1 --logging 0 --method dopri5 --time 8  --no_early --coupling_strength 1 --split_rate 1 --dataset Citeseer
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset Cora
done
done
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset Citeseer
done
done
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset Pubmed
done
done
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset Computers
done
done
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 1 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset CoauthorCS
done
done
done
done

for split in 1 2 5 10 20; do
for coup in 0.7 0.8 0.9 1; do
for time in 2 3 5 7 8 10 12 16 18; do
for run in 1 2 3 4; do
    python3 run_GNN.py --kuramoto 1 --logging 0 --method dopri5 --time $time --no_early --coupling_strength $coup --split_rate $split --dataset Photo
done
done
done
done

# done
# done
# for time in 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --coupling_strength 2 --split_rate $split --time $time --method dopri5 --kuramoto 1 --dataset Citeseer
# done
# done
# done

# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --split_rate $split --dataset CoauthorCS
# done
# done

# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --split_rate $split --dataset Photo
# done
# done
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --split_rate $split --dataset Computers
# done
# done


# for time in 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --coupling_strength 2 --split_rate $split --time $time --method dopri5 --kuramoto 1 --dataset CoauthorCS
# done
# done
# done


# for time in 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --coupling_strength 2 --split_rate $split --time $time --method dopri5 --kuramoto 1 --dataset Photo
# done
# done
# done

# for time in 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --coupling_strength 2 --split_rate $split --time $time --method dopri5 --kuramoto 1 --dataset Computers
# done
# done
# done

# for time in 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
# for split in 1 2 5 10 20; do
# for run in 1 2 3 4 5; do
#     python3 run_GNN.py --run_time $run --coupling_strength 2 --split_rate $split --time $time --method dopri5 --kuramoto 1 --dataset Pubmed
# done
# done
# done