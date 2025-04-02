function main { python scripts/gen-pong/main.py --silent $@ ; }
function eval { python scripts/gen-pong/eval.py --silent $@ ; }

for seed in $(seq 1 5)
do

echo
echo "running id ..."
main --seed $seed --env Train --exp-name id$seed
eval --seed $seed --env Train --exp-name id$seed --output training
eval --seed $seed --env TestStochastic --exp-name id$seed --output test_stochastic
main --seed $seed --env TestStochastic --exp-name id$seed-retrained-stochastic --retrain id$seed
eval --seed $seed --env TestStochastic --exp-name id$seed-retrained-stochastic --output test_stochastic
eval --seed $seed --env TestObservational --exp-name id$seed --output test_observational
main --seed $seed --env TestObservational --exp-name id$seed-retrained-observational --retrain id$seed
eval --seed $seed --env TestObservational --exp-name id$seed-retrained-observational --output test_observational

echo
echo "running justfield ..."
main --seed $seed --filter JustField --env Train --exp-name justfield$seed
eval --seed $seed --filter JustField --env Train --exp-name justfield$seed --output training
eval --seed $seed --filter JustField --env TestStochastic --exp-name justfield$seed --output test_stochastic
main --seed $seed --filter JustField --env TestStochastic --exp-name justfield$seed-retrained-stochastic --retrain justfield$seed
eval --seed $seed --filter JustField --env TestStochastic --exp-name justfield$seed-retrained-stochastic --output test_stochastic
eval --seed $seed --filter JustField --env TestObservational --exp-name justfield$seed --output test_observational
main --seed $seed --filter JustField --env TestObservational --exp-name justfield$seed-retrained-observational --retrain justfield$seed
eval --seed $seed --filter JustField --env TestObservational --exp-name justfield$seed-retrained-observational --output test_observational

echo
echo "running justcount ..."
main --seed $seed --filter JustCount --env Train --exp-name justcount$seed
eval --seed $seed --filter JustCount --env Train --exp-name justcount$seed --output training
eval --seed $seed --filter JustCount --env TestStochastic --exp-name justcount$seed --output test_stochastic
main --seed $seed --filter JustCount --env TestStochastic --exp-name justcount$seed-retrained-stochastic --retrain justcount$seed
eval --seed $seed --filter JustCount --env TestStochastic --exp-name justcount$seed-retrained-stochastic --output test_stochastic
eval --seed $seed --filter JustCount --env TestObservational --exp-name justcount$seed --output test_observational
main --seed $seed --filter JustCount --env TestObservational --exp-name justcount$seed-retrained-observational --retrain justcount$seed
eval --seed $seed --filter JustCount --env TestObservational --exp-name justcount$seed-retrained-observational --output test_observational

echo
echo "running dists ..."
main --seed $seed --filter FieldDistractions --env Train --exp-name dists$seed
eval --seed $seed --filter FieldDistractions --env Train --exp-name dists$seed --output training
eval --seed $seed --filter FieldDistractions --env TestStochastic --exp-name dists$seed --output test_stochastic
main --seed $seed --filter FieldDistractions --env TestStochastic --exp-name dists$seed-retrained-stochastic --retrain dists$seed
eval --seed $seed --filter FieldDistractions --env TestStochastic --exp-name dists$seed-retrained-stochastic --output test_stochastic
eval --seed $seed --filter FieldDistractions --env TestObservational --exp-name dists$seed --output test_observational
main --seed $seed --filter FieldDistractions --env TestObservational --exp-name dists$seed-retrained-observational --retrain dists$seed
eval --seed $seed --filter FieldDistractions --env TestObservational --exp-name dists$seed-retrained-observational --output test_observational

done # seed
