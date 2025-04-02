function main { python scripts/gen-minigrid/main.py --silent $@ ; }
function eval { python scripts/gen-minigrid/eval.py --silent $@ ; }

for seed in $(seq 1 5)
do

echo
echo "running id ..."
main --seed $seed --env MultiCoupled --exp-name id$seed
eval --seed $seed --env MultiCoupled --exp-name id$seed --output training
eval --seed $seed --env Single --exp-name id$seed
main --seed $seed --env Single --exp-name id$seed-retrained --retrain id$seed
eval --seed $seed --env Single --exp-name id$seed-retrained

echo
echo "running hidecols ..."
main --seed $seed --filter HideCols --env MultiCoupled --exp-name hidecols$seed
eval --seed $seed --filter HideCols --env MultiCoupled --exp-name hidecols$seed --output training
eval --seed $seed --filter HideCols --env Single --exp-name hidecols$seed
main --seed $seed --filter HideCols --env Single --exp-name hidecols$seed-retrained --retrain hidecols$seed
eval --seed $seed --filter HideCols --env Single --exp-name hidecols$seed-retrained

echo
echo "running hidedoor ..."
main --seed $seed --filter HideDoor --env MultiCoupled --exp-name hidedoor$seed
eval --seed $seed --filter HideDoor --env MultiCoupled --exp-name hidedoor$seed --output training
eval --seed $seed --filter HideDoor --env Single --exp-name hidedoor$seed
main --seed $seed --filter HideDoor --env Single --exp-name hidedoor$seed-retrained --retrain hidedoor$seed
eval --seed $seed --filter HideDoor --env Single --exp-name hidedoor$seed-retrained

echo
echo "running hideboth ..."
main --seed $seed --filter HideBoth --env MultiCoupled --exp-name hideboth$seed
eval --seed $seed --filter HideBoth --env MultiCoupled --exp-name hideboth$seed --output training
eval --seed $seed --filter HideBoth --env Single --exp-name hideboth$seed
main --seed $seed --filter HideBoth --env Single --exp-name hideboth$seed-retrained --retrain hideboth$seed
eval --seed $seed --filter HideBoth --env Single --exp-name hideboth$seed-retrained

echo
echo "running onehot ..."
main --seed $seed --filter OneHot --env MultiCoupled --exp-name onehot$seed
eval --seed $seed --filter OneHot --env MultiCoupled --exp-name onehot$seed --output training
eval --seed $seed --filter OneHot --env Single --exp-name onehot$seed
main --seed $seed --filter OneHot --env Single --exp-name onehot$seed-retrained --retrain onehot$seed
eval --seed $seed --filter OneHot --env Single --exp-name onehot$seed-retrained

done # seed
