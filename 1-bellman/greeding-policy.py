import numpy as np
import copy


p = [0.8, 0.5, 1.0]
gamma = 0.95

r = np.zeros((3, 3, 2))
r[0, 1, 0] = 1.0
r[0, 2, 0] = 2.0
r[0, 0, 1] = 0
r[1, 0, 0] = 1.0
r[1, 2, 0] = 2.0
r[1, 1, 1] = 1.0
r[2, 0, 0] = 1.0
r[2, 1, 0] = 0.0
r[2, 2, 1] = -1.0


# value function
v = [0, 0, 0]
v_prev = copy.copy(v)

# value iteration
q = np.zeros((3, 2))

# 方策分布初期化
pi = [0.5, 0.5, 0.5]

# 方策評価関数
def policy_evaluation(pi, p, r, gamma):
    # init
    R = [0, 0, 0]
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))

    for i in range(3):
        P[i, i] = 1 - pi[i]
        P[i, (i + 1) % 3] = p[i] * pi[i]
        P[i, (i + 2) % 3] = (1 - p[i]) * pi[i]

        # 報酬ベクトル
        R[i] = pi[i] * (pi[i] * r[i, (i+1)%3,0]+ \
                (1-pi[i]) * r[i, (i+2)%3,0]) + \
                (1-pi[i]) * r[i, i, 1]
        
        # ベルマンの解
        A = np.eye(3) - gamma * P
        B = np.linalg.inv(A)
        v_sol = np.dot(B, R)
    return v_sol