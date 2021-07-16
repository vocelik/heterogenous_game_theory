from .enums import Action
import numpy as np

C, D = Action.C, Action.D

def cooperate(*args):
    # Cooperates unconditionally
    return C
cooperate.name = 'cooperate'

def defect(*args):
    # Defects unconditionally
    return D
defect.name = 'defect'

def tit_for_tat(selfmoves, othermoves):
    # Cooperates on the first round and imitates its
    # opponent's previous move thereafter.
    if othermoves == []:
        return C
    else:
        return othermoves[-1]
tit_for_tat.name = 'tit_for_tat'

def generous_tit_for_tat(selfmoves, othermoves):
    # Cooprates on the first round and after its opponent cooperates.
    # Following a defection,it cooperates with probability .7
    if selfmoves == []:
        return C
    elif othermoves[-1] == C:
        return C
    else:
        return np.random.choice([C, D], p=[0.3, 0.7])
generous_tit_for_tat.name = 'generous_tit_for_tat'

def win_stay_lose_shift(selfmoves, othermoves):
    # Cooperates if it and its opponent moved alike in previous move
    # and defects if they moved differently.
    if selfmoves == []:
        return C
    elif othermoves[-1] == selfmoves[-1]:
        return C
    else:
        return D
win_stay_lose_shift.name = 'win_stay_lose_shift'

def tit_for_two_tats(selfmoves, othermoves):
    # Cooperates unless defected against twice in a row
    if len(selfmoves) < 2:
        return C
    else:
        return D if othermoves[-2:] == [D,D] else C
tit_for_two_tats.name = 'tit for two tats'

def APavlov(selfmoves, othermoves):    
    # Employs TFT for the first six rounds, places opponent into
    # one of five categories according to its responses and
    # plays an optimal strategy for each.

    # it does not work yet because strategies are not defined as classes.

    opponent_class = None

    if selfmoves == []:
        return C
    # first six rounds TFT
    if len(selfmoves) < 6:
        return D if othermoves[-1] == D else C
    # after six rounds move on to classification
    if len(selfmoves) % 6 == 0:
        # classify opponent
        if othermoves[-6:] == [C] * 6:
            opponent_class = "COOP"
            print("COOP")
            print(othermoves[-6:])
        if othermoves[-6:].count(D) >= 4:
            opponent_class = "ALLD"
            print("ALLD")
            print(othermoves[-6:])
        if othermoves[-6:].count(D) == 3:
            opponent_class = "STFT"
            print("STFT")
            print(othermoves[-6:])
        if not opponent_class:
            opponent_class = "RANDOM"
            print("RANDOM")
            print(othermoves[-6:])
    # play according to classification
    if opponent_class in ["RANDOM", "ALLD"]:
        return D
    if opponent_class == "STFT":
        # TFTT
        return D if othermoves[-2:] == [D, D] else C
    if opponent_class == "COOP":
        # TFT
        return D if othermoves[-1] == [D] else C  

    print("nothing was returned")

APavlov.name = 'Adaptive Pavlov'