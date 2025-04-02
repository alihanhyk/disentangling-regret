function main { python scripts/regret/main.py --silent $@ ; }
function eval { python scripts/regret/eval.py --silent $@ ; }

for seed in $(seq 1 5)
do

### MINIGRID

echo
echo "running minigrid/opt ..."
main --env MiniGrid --seed $seed --exp-name opt_$seed
eval --env MiniGrid --seed $seed --exp-name opt_$seed

tags=(10 20 30 40 50)
vals=(0.1 0.2 0.3 0.4 0.5)
for i in $(seq 0 4)
do
    echo
    echo "running minigrid/rec${tags[@]:i:1} ..."
    main --env MiniGrid --seed $seed --exp-name rec${tags[@]:i:1}_$seed --resave opt_$seed --p-obs ${vals[@]:i:1}
    eval --env MiniGrid --seed $seed --exp-name rec${tags[@]:i:1}_$seed
    main --env MiniGrid --seed $seed --exp-name rec${tags[@]:i:1}R_$seed --retrain rec${tags[@]:i:1}_$seed
    eval --env MiniGrid --seed $seed --exp-name rec${tags[@]:i:1}R_$seed
done

tags=(10 20 30 40 50)
vals=(0.1 0.2 0.3 0.4 0.5)
for i in $(seq 0 4)
do
    echo
    echo "running minigrid/dec${tags[@]:i:1} ..."
    main --env MiniGrid --seed $seed --exp-name dec${tags[@]:i:1}_$seed --p-action ${vals[@]:i:1}
    eval --env MiniGrid --seed $seed --exp-name dec${tags[@]:i:1}_$seed
    main --env MiniGrid --seed $seed --exp-name dec${tags[@]:i:1}R_$seed --retrain dec${tags[@]:i:1}_$seed
    eval --env MiniGrid --seed $seed --exp-name dec${tags[@]:i:1}R_$seed
done

done # seed

### PONG

for seed in $(seq 1 9)
do

echo
echo "running pong/opt ..."
main --env Pong --seed $seed --exp-name opt_$seed
eval --env Pong --seed $seed --exp-name opt_$seed

tags=(10 20 30 40 50)
vals=(0.1 0.2 0.3 0.4 0.5)
for i in $(seq 0 4)
do
    echo
    echo "running pong/rec${tags[@]:i:1} ..."
    main --env Pong --seed $seed --exp-name rec${tags[@]:i:1}_$seed --resave opt_$seed --p-obs ${vals[@]:i:1}
    eval --env Pong --seed $seed --exp-name rec${tags[@]:i:1}_$seed
    main --env Pong --seed $seed --exp-name rec${tags[@]:i:1}R_$seed --retrain rec${tags[@]:i:1}_$seed
    eval --env Pong --seed $seed --exp-name rec${tags[@]:i:1}R_$seed
done

tags=(10 20 30 40 50)
vals=(0.1 0.2 0.3 0.4 0.5)
for i in $(seq 0 4)
do
    echo
    echo "running pong/dec${tags[@]:i:1} ..."
    main --env Pong --seed $seed --exp-name dec${tags[@]:i:1}_$seed --p-action ${vals[@]:i:1}
    eval --env Pong --seed $seed --exp-name dec${tags[@]:i:1}_$seed
    main --env Pong --seed $seed --exp-name dec${tags[@]:i:1}R_$seed --retrain dec${tags[@]:i:1}_$seed
    eval --env Pong --seed $seed --exp-name dec${tags[@]:i:1}R_$seed
done

done # seed
