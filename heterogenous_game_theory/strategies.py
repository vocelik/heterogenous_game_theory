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

    opponent_class = None

    if len(selfmoves) < 1:
        print("THIS WORKS")
        return C
    # first six rounds TFT
    if len(selfmoves) < 6:
        return D if othermoves[-1] == D else C
    # after six rounds move on to classification
    if len(selfmoves) > 6:
        # classify opponent
        if othermoves[-6:] == [D] * 6:
            opponent_class = "COOP"
        if othermoves[-6:].count(D) >= 4:
            opponent_class = "ALLD"
        if othermoves[-6:].count(D) == 3:
            opponent_class = "STFT"
        if not opponent_class:
            opponent_class = "Random"
    # play according to classification
    if opponent_class in ["Random", "ALLD"]:
        return D
    if opponent_class == "STFT":
        # TFTT
        return D if othermoves[-2:] == [D, D] else C
    if opponent_class == "Cooperative":
        # TFT
        return D if othermoves[-1:] == [D] else C  

APavlov.name = 'Adaptive Pavlov'