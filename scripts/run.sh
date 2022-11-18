# if Pretrained
#   if No Replay
echo "NoPretrained-NoReplay-NoCoreset-NoArgument"
python3 ../main.py --replay=False --results_dir="../results/NoPretrained-NoReplay-NoCoreset-NoArgument/"
#   else
#       if No Coreset
echo "NoPretrained-Replay-NoCoreset-NoArgument"
python3 ../main.py --replay=True --results_dir="../results/NoPretrained-Replay-NoCoreset-NoArgument/"
#       else
#           if No Argument
echo "NoPretrained-Replay-Coreset-NoArgument"
python3 ../main.py --replay=True --coreset=True --results_dir="../results/NoPretrained-Replay-Coreset-NoArgument/"
#           else
echo "NoPretrained-Replay-Coreset-Argument"
python3 ../main.py --replay=True --coreset=True --argument=True --results_dir="../results/NoPretrained-Replay-Coreset-Argument/"
# else
#   if No Replay
echo "Pretrained-NoReplay-NoCoreset-NoArgument"
python3 ../main.py --pretrained=True --replay=False --results_dir="../results/Pretrained-NoReplay-NoCoreset-NoArgument/"
#   else
#       if No Coreset
echo "Pretrained-Replay-NoCoreset-NoArgument"
python3 ../main.py --pretrained=True --replay=True --results_dir="../results/Pretrained-Replay-NoCoreset-NoArgument/"
#       else
#           if No Argument
echo "Pretrained-Replay-Coreset-NoArgument"
python3 ../main.py --pretrained=True --replay=True --coreset=True --results_dir="../results/Pretrained-Replay-Coreset-NoArgument/"
#           else
echo "Pretrained-Replay-Coreset-Argument"
python3 ../main.py --pretrained=True --replay=True --coreset=True --argument=True --results_dir="../results/Pretrained-Replay-Coreset-Argument/"
