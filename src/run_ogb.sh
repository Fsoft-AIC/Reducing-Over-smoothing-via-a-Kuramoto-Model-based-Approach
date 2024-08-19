for run in 1 2 3 4 5 6 7 8 9 10; do
   for coup in 0.5 0.7 0.9; do
      for time in  3.5 3.6760155951687636 3.7 4 4.6; do
         python3 run_GNN.py --coupling_strength $coup --time $time --method dopri5 --dataset ogbn-arxiv --run_time $run 
      done
   done
done
