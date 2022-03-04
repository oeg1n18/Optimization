import numpy as np
import copy


class LinProg:
    def __init__(self, C, A, b):
        self.A = A
        self.C = C
        self.b = b
        self.tab0 = np.concatenate((np.concatenate(((-self.A), self.b), axis=1), C), axis=0)
        self.tab1 = np.concatenate((np.concatenate(((np.zeros(self.A.shape)), self.b), axis=1), C), axis=0)

        self.n_V = C.size
        self.n_sV = self.tab0.shape[1] - 1

        self.nB = np.linspace(0, self.n_V - 1, self.n_V)
        self.B = np.linspace(self.n_V, self.n_V + self.n_sV - 1, self.n_sV)
        self.pivot = [0, 0]

    def get_pivot_column(self):
        return np.argmax(self.tab0[self.tab0.shape[0] - 1, :-1])

    def get_pivot(self):
        j = self.get_pivot_column()
        indices = [i for i in range(self.n_V) if self.tab0[i, j] < 0]
        indices = np.array(indices)

        ratio_max = -np.inf
        r_i_max = 0
        for r_i in indices:
            ratio = self.tab0[r_i, self.n_V - 1] / self.tab0[r_i, j]
            if ratio > ratio_max:
                r_i_max = r_i
                ratio_max = ratio

        self.B[r_i_max] = self.nB[j]
        self.nB[j] = self.B[r_i_max]
        self.pivot = [r_i_max, j]

    def update_row(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        for j in range(self.n_V):
            if j != j_p:
                self.tab1[i_p, j] = - self.tab0[i_p, j] / self.tab0[i_p, j_p]
            else:
                self.tab1[i_p, j_p] = 1 / self.tab0[i_p, j_p]

    def update_col(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        for i in range(self.tab0.shape[0]):
            if i != i_p:
                if self.tab0[i_p, j_p] == 0:
                    pass
                else:
                    self.tab1[i, j_p] = self.tab0[i, j_p] / self.tab0[i_p, j_p]

    def update_rest(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        indices = []
        for i in range(self.tab0.shape[0]):
            for j in range(self.tab0.shape[1]):
                if i != i_p:
                    if j != j_p:
                        indices.append([i, j])
        for ind in indices:
            [i_p, j_p] = [self.pivot[0], self.pivot[1]]
            # old - new * old
            old = self.tab0[ind[0], ind[1]]
            new = self.tab1[ind[0], j_p]
            old2 = self.tab0[i_p, ind[1]]
            self.tab1[ind[0], ind[1]] = old - new * old2


class Phase_1:
    def __init__(self, C1, A1, b1, A2, b2):
        self.C = C1
        self.A1 = A1
        self.A2 = A2
        self.b1 = b1
        self.b2 = b2
        self.A = np.concatenate((A1, -A2), axis=0)
        self.b = np.concatenate((-self.b1, self.b2), axis=0)
        self.x0 = np.concatenate((np.ones([self.b1.shape[0], 1]), np.zeros([self.b2.shape[0], 1])), axis=0)
        self.A = np.concatenate((self.A, self.x0), axis=1)
        self.Z = np.concatenate((self.C[:-1], [0], self.C[-1:]))
        self.z0 = np.zeros([self.A.shape[1] + 1])
        self.z0[self.A.shape[1] - 1] = -1
        self.tab0 = np.concatenate((self.A, self.b.reshape(self.b.size, 1)), axis=1)
        self.tab0 = np.concatenate((self.tab0, self.Z.reshape((1, self.Z.size))), axis=0)
        self.tab0 = np.concatenate((self.tab0, self.z0.reshape((1, self.z0.size))), axis=0)
        self.tab1 = copy.copy(self.tab0)
        self.pivot = [0, 0]
        self.n_V = self.tab0.shape[1]
        self.nB = np.append(np.arange(1, self.tab0.shape[1] - 1), [0])
        self.B = np.arange(self.nB[-2] + 1, self.nB[-2] + self.tab0.shape[0] - 1)

    def get_initial_pivot(self):
        i = np.argmin(self.tab0[:, self.tab0.shape[1] - 1])
        j = self.tab0.shape[1] - 2
        self.pivot = [i, j]

    def get_pivot_column(self):
        return np.argmax(self.tab0[self.tab0.shape[0] - 1, :-1])

    def get_pivot(self):
        j = self.get_pivot_column()
        indices = [i for i in range(self.n_V) if self.tab0[i, j] < 0]
        indices = np.array(indices)
        ratio_max = -np.inf
        r_i_max = 0
        for r_i in indices:
            ratio = self.tab0[r_i, self.n_V - 1] / self.tab0[r_i, j]
            if ratio > ratio_max:
                r_i_max = r_i
                ratio_max = ratio

        self.B[r_i_max] = self.nB[j]
        self.nB[j] = self.B[r_i_max]
        self.pivot = [r_i_max, j]

    def update_row(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        for j in range(self.n_V):
            if j != j_p:
                self.tab1[i_p, j] = - self.tab0[i_p, j] / self.tab0[i_p, j_p]
            else:
                self.tab1[i_p, j_p] = 1 / self.tab0[i_p, j_p]

    def update_col(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        for i in range(self.tab0.shape[0]):
            if i != i_p:
                if self.tab0[i_p, j_p] == 0:
                    pass
                else:
                    self.tab1[i, j_p] = self.tab0[i, j_p] / self.tab0[i_p, j_p]

    def update_rest(self):
        [i_p, j_p] = [self.pivot[0], self.pivot[1]]
        indices = []
        for i in range(self.tab0.shape[0]):
            for j in range(self.tab0.shape[1]):
                if i != i_p:
                    if j != j_p:
                        indices.append([i, j])
        for ind in indices:
            [i_p, j_p] = [self.pivot[0], self.pivot[1]]
            # old - new * old
            old = self.tab0[ind[0], ind[1]]
            new = self.tab1[ind[0], j_p]
            old2 = self.tab0[i_p, ind[1]]
            self.tab1[ind[0], ind[1]] = old - new * old2


class LinProg_2Phase(LinProg, Phase_1):
    def __init__(self, C1, A1, b1, A2, b2):
        self.phase1 = Phase_1(C1, A1, b1, A2, b2)
        self.phase1_complete_flag = 0
        self.step = 0
        self.solution = np.zeros([self.phase1.B.size + self.phase1.nB.size, 1])
        self.objective = 0
        self.phase2 = 0

    def solve(self):
        self.solve_phase1()
        print(self.phase1.tab0)
        print(self.phase1.nB)
        print(self.phase1.B)

    def solve_phase1(self):
        while self.phase1_complete_flag == 0:
            print(self.phase1.tab0)
            print(self.phase1.pivot)
            self.phase1 = self.jordan_exchange(self.phase1)
            if self.phase1.tab0[self.phase1.tab0.shape[0] - 1, self.phase1.tab0.shape[0] - 1] == 0 and self.step > 0:
                self.phase1_complete_flag = 1
        return 1

    def jordan_exchange(self, prog):
        if self.step == 0:
            prog.get_initial_pivot()
        else:
            prog.get_pivot()
        print(prog.nB)
        print(prog.B)
        prog.update_col()
        prog.update_row()
        prog.update_rest()
        prog.tab0 = copy.copy(prog.tab1)
        prog.B[prog.pivot[0]] = prog.nB[prog.pivot[1]]
        prog.nB[prog.pivot[1]] = prog.B[prog.pivot[0]]
        self.step += 1
        return prog

    def test_unbounded(self):
        for i in range(self.prog.tab0.shape[1] - 1):
            if sum(v >= 0 for v in self.prog.tab0[:, i]) == self.prog.tab0.shape[0]:
                return 1

    def test_optimal(self):
        tab = self.prog.tab0
        obj_arr = tab[-1:, :][0, :]
        rhs_arr = tab[:, self.prog.n_V - 1]
        if sum(n > 0 for n in obj_arr[:-1]) == 0:
            if sum(n < 0 for n in rhs_arr) == 0:
                return 1
        else:
            return 0

    def test_infeasible(self):
        tab = self.prog.tab1
        obj_arr = tab[self.prog.n_V - 1, :]
        rhs_arr = tab[:, self.prog.n_V - 1]
        if sum(n > 0 for n in obj_arr) == 1:
            if sum(n < 0 for n in rhs_arr) > 0:
                return 1
        else:
            return 0

    def test_infinite_solution(self):
        if sum(n == 0 for n in self.prog.tab0[-1:, :][0]) > 0:
            return 1
