import omok
import numpy as np


def transpose(state: np.ndarray, move: int):
    state_a = state.T
    y, x = divmod(move, 15)
    move_a = x * 15 + y
    return state_a, move_a


def rotate(state: np.ndarray, move: int):
    state_a = np.rot90(state)
    y, x = divmod(move, 15)
    move_a = (14 - x) * 15 + y
    return state_a, move_a


def augment(state: np.ndarray, move: int):
    states = []
    moves = []

    s, m = state, move
    states.append(s)
    moves.append(m)

    for _ in range(3):
        s, m = rotate(s, m)
        states.append(s)
        moves.append(m)

    s, m = transpose(state, move)
    states.append(s)
    moves.append(m)

    for _ in range(3):
        s, m = rotate(s, m)
        states.append(s)
        moves.append(m)

    return states, moves


def check(state: np.ndarray, move: int):
    s = state.copy()
    s = s.reshape(-1)
    s[move] = 9
    s = s.reshape(15, 15)
    print(s, move)


def test():
    env = omok.Omok()

    moves = [0, 1, 2, 3, 223, 224]
    for move in moves:
        env(move)

    state = env.get_state()
    move = 4
    state_t, move_t = transpose(state, move)
    state_r, move_r = rotate(state, move)

    check(state, move)
    check(state_t, move_t)
    check(state_r, move_r)


def test2():
    env = omok.Omok()

    moves = [0, 1, 2, 3, 223, 224]
    for move in moves:
        env(move)

    state = env.get_state()
    move = 4

    states, moves = augment(state, move)

    print(states.shape, moves.shape)

    for i in range(len(states)):
        check(states[i], moves[i])


if __name__ == '__main__':
    test()
    test2()
