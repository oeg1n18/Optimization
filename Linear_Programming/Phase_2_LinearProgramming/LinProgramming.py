import numpy as np
import copy as copy


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


class LinProg_phase2(LinProg):
    def __init__(self, C, A, b):
        super().__init__(C, A, b)
        self.prog = LinProg(C, A, b)
        self.step = 0
        self.solution = np.zeros([self.prog.B.size + self.prog.nB.size + 1, 1])
        self.objective = 0

    def solve(self):
        i = 0
        while 1:
            self.jordan_exchange()
            if self.test_infinite_solution() == 1:
                print("This problem has infinite solutions")
                print("step reached: " + str(self.step))
                break
            if self.test_optimal() == 1:
                print("you have reached the optimal Tabular")
                print(str(self.step) + " steps were taken")
                break
            if self.test_unbounded() == 1:
                print("This Problem Is Unbounded")
                print("step reached: " + str(self.step))
                break
            if self.test_infeasible() == 1:
                print("This problem is infeasible")
                print("step reached: " + str(self.step))
                break

        rhs_j = self.prog.tab0.shape[1] - 1
        for i, rhs_i in zip(self.prog.B, range(self.prog.B.size)):
            self.solution[int(i)] = self.prog.tab0[rhs_i, rhs_j]
        self.objective = self.prog.tab0[self.prog.tab0.shape[0] - 1, self.prog.tab0.shape[1] - 1]

    def jordan_exchange(self):
        self.prog.get_pivot()
        self.prog.update_col()
        self.prog.update_row()
        self.prog.update_rest()
        self.prog.tab0 = copy.copy(self.prog.tab1)
        self.prog.tab1[0:-1, 0:-1] = 0
        self.step += 1

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
