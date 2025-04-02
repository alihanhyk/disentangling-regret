function main { python scripts/gen-minigrid/main.py $@ ; }
function eval { python scripts/gen-minigrid/eval.py $@ ; }

for seed in $(seq 1 10)
do
    for size in 1 2 3 4 6 8 12 16 24 32 48 64
    do
        main --seed $seed --rep-size $size --env MultiCoupled --exp-name use$seed-repsize$size
        eval --seed $seed --rep-size $size --env MultiCoupled --exp-name use$seed-repsize$size --output training
        eval --seed $seed --rep-size $size --env Single --exp-name use$seed-repsize$size
        main --seed $seed --rep-size $size --env Single --exp-name use$seed-repsize$size-retrained --retrain use$seed-repsize$size
        eval --seed $seed --rep-size $size --env Single --exp-name use$seed-repsize$size-retrained
    done
done
