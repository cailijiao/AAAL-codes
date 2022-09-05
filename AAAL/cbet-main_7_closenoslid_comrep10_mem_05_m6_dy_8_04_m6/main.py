from src.arguments import parser

from src.algos.torchbeast import train as train_vanilla
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity
from src.algos.rnd import train as train_rnd
from src.algos.cbet import train as train_cbet
from src.algos.ride import train as train_ride

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def main(flags):
    if flags.model == 'vanilla':
        train_vanilla(flags)
    elif flags.model == 'count':
        train_count(flags)
    elif flags.model == 'curiosity':
        train_curiosity(flags)
    elif flags.model == 'rnd':
        train_rnd(flags)
    elif flags.model == 'ride':
        train_ride(flags)
    elif flags.model == 'cbet':
        train_cbet(flags)
    else:
        raise NotImplementedError("This model has not been implemented. "\
        "The available options are: cbet, vanilla, count, curiosity, rnd, ride.")

'''
OMP_NUM_THREADS=1 python main.py --model cbet --env 
'MiniGrid-Unlock-v0','MiniGrid-DoorKey-5x5-v0','MiniGrid-KeyCorridorS3R3-v0','MiniGrid-UnlockPickup-v0','MiniGrid-BlockedUnlockPickup-v0',
'MiniGrid-MultiRoom-N6-v0','MiniGrid-MultiRoom-N12-S10-v0','MiniGrid-ObstructedMaze-1Dlh-v0','MiniGrid-ObstructedMaze-2Dlh-v0','MiniGrid-ObstructedMaze-2Dlhb-v0'
 --no_reward --intrinsic_reward_coef=0.005

'''
'''
OMP_NUM_THREADS=1 python main.py --model cbet --env 'MiniGrid-ObstructedMaze-2Dlhb-v0' --intrinsic_reward_coef=0.0 --checkpoint=./logs/cbet-20220501-032730/model.tar 
--total_frames=???12000000
'''
'''
OMP_NUM_THREADS=1 python main.py --model cbet --env 'MiniGrid-Unlock-v0' --intrinsic_reward_coef=0.005 --total_frames=5000000
'''
if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)
