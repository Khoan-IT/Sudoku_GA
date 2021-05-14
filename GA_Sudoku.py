import numpy as np
import random
import operator

from past.builtins import range

random.seed()

Nd = 9  # size of game

class Population(object):

    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []

        # Tiềm kiếm những số có thể điền được ở từng vị trí
        helper = Candidate()
        helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if ((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Nếu vị trí bằng không và thõa mãn luật thì số đó có thể là lời giải
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        # Nếu là số đề cho thì giữ nguyên
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Sinh ra quần thể
        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):  # Tạo từng hàng của 1 ứng viên
                row = np.zeros(Nd)
                for j in range(0, Nd):

                    # Nếu là số đề cho thì giữ nguyên
                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    # Nếu chưa có thì chọn ngẫu nhiên 1 số trong những số có thể điền
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                # Nếu không tìm được thì lặp lại 200000 để cố gắng tìm ra 1 hàng hợp lý
                ii = 0
                while len(list(set(row))) != Nd:
                    ii += 1
                    if ii > 200000:
                        return 0
                    for j in range(0, Nd):
                        if given.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row
            self.candidates.append(g)
        # Tính toán độ tương thích của toàn bộ quần thế
        self.update_fitness()
        return 1

    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return

    def sort(self):
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return

class Candidate(object):

    def __init__(self):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        return

    def update_fitness(self):
        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        column_sum = 0
        block_sum = 0

        self.values = self.values.astype(int)
        # Tính toán cho mỗi cột
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.values[i][j] - 1] += 1
            for k in range(len(column_count)):
                if column_count[k] == 1:
                    # Mỗi cột có Nd ô và 1 game có Nd cột nên sẽ là 1/Nd/Nd
                    column_sum += (1/Nd)/Nd
            column_count = np.zeros(Nd)

        # Tính toán cho mỗi khối 3x3
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j] - 1] += 1
                block_count[self.values[i][j + 1] - 1] += 1
                block_count[self.values[i][j + 2] - 1] += 1

                block_count[self.values[i + 1][j] - 1] += 1
                block_count[self.values[i + 1][j + 1] - 1] += 1
                block_count[self.values[i + 1][j + 2] - 1] += 1

                block_count[self.values[i + 2][j] - 1] += 1
                block_count[self.values[i + 2][j + 1] - 1] += 1
                block_count[self.values[i + 2][j + 2] - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1/Nd)/Nd
                block_count = np.zeros(Nd)

        # Tính toán độ tương thích cho tổng thể giữa các cột và các khối
        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness
        return

    def mutate(self, mutation_rate, given):

        r = random.uniform(0, 1.1)
        while r > 1:  # Nếu lớn hơn 1 thì sinh lại 1 số mới. Sử dụng phân phối đều liên tục
            r = random.uniform(0, 1)

        success = False
        if r < mutation_rate:  # TÌm vị trí để gây đột biến
            while not success:
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1

                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                    # Kiểm tra vị trí đột biến
                if given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0:
                    if not given.is_column_duplicate(to_column, self.values[row1][from_column]) and not given.is_column_duplicate(from_column, self.values[row2][to_column]) and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column]):
                        # Nếu thõa mãn thì thay đổi giá trị
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True

        return success


class Fixed(Candidate):

    def __init__(self, values):
        self.values = values
        return

    def is_row_duplicate(self, row, value):
        for column in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, column, value):
        for row in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_block_duplicate(self, row, column, value):
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
            or (self.values[i][j + 1] == value)
            or (self.values[i][j + 2] == value)
            or (self.values[i + 1][j] == value)
            or (self.values[i + 1][j + 1] == value)
            or (self.values[i + 1][j + 2] == value)
            or (self.values[i + 2][j] == value)
            or (self.values[i + 2][j + 1] == value)
            or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False

    def make_index(self, v):
        if v <= 2:
            return 0
        elif v <= 5:
            return 3
        else:
            return 6

    def no_duplicates(self):
        for row in range(0, Nd):
            for col in range(0, Nd):
                if self.values[row][col] != 0:

                    cnt1 = list(self.values[row]).count(self.values[row][col])
                    cnt2 = list(self.values[:,col]).count(self.values[row][col])

                    block_values = [y[self.make_index(col):self.make_index(col)+3] for y in
                                    self.values[self.make_index(row):self.make_index(row)+3]]
                    block_values_ = [int(x) for y in block_values for x in y]
                    cnt3 = block_values_.count(self.values[row][col])

                    if cnt1 > 1 or cnt2 > 1 or cnt3 > 1:
                        return False
        return True

class Tournament(object):
    def __init__(self):
        return

    def compete(self, candidates):
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        if (f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.80
        r = random.uniform(0, 1.1)
        while (r > 1):
            r = random.uniform(0, 1.1)
        if (r < selection_rate):
            return fittest
        else:
            return weakest


class CycleCrossover(object):

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossover_rate):
        child1 = Candidate()
        child2 = Candidate()

        # Sao chép gene từ cha mẹ sang con
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while (r > 1):
            r = random.uniform(0, 1.1)

        if (r < crossover_rate):
            # Chọn các hàng xảy ra sự trao đổi.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while (crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)

            if (crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2):
        child_row1 = np.zeros(Nd)
        child_row2 = np.zeros(Nd)

        remaining = range(1, Nd + 1)
        cycle = 0

        while ((0 in child_row1) and (0 in child_row2)):  # Khi con chưa được hoàn thành
            if (cycle % 2 == 0):
                # Gán giá trị trực tiếp từ cha mẹ sang con
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while (next != start):  
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:
                # Trao đổi chéo từ cha sang child2 và mẹ sang child1
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while (next != start): 
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]

                cycle += 1

        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if (parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if (parent_row[i] == value):
                return i


class Sudoku(object):
    def __init__(self):
        self.given = None
        return

    def load(self, p):
        self.given = Fixed(p)
        return

    def solve(self):

        Nc = 1000  # Kích thước quần thể
        Ne = int(0.05 * Nc)  # Số lượng nhân tố
        Ng = 10000  # Số lần lai
        Nm = 0  # Số đột biến

        # Tham số đột biến
        mutation_rate = 0.06

        if self.given.no_duplicates() == False:
            return (-1, 1)
        self.population = Population()
        print("Khởi tạo quần thể")
        if self.population.seed(Nc, self.given) ==  1:
            pass
        else:
            return (-1, 1)

        for generation in range(0, Ng): 
            best_fitness = 0.0
            for c in range(0, Nc):
                fitness = self.population.candidates[c].fitness
                if (fitness == 1):
                    print("Bài toán được giải ở lần lai: %d!" % generation)
                    return (generation, self.population.candidates[c])

                if (fitness > best_fitness):
                    best_fitness = fitness

            print("Lần lai:", generation, "Mức độ phù hợp:", best_fitness)
            # Tạo quần thể mới
            next_population = []

            # Chọn những các thể để làm thế hệ tiếp theo
            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = np.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Chọn cá thể để lai
            for count in range(Ne, Nc, 2):
                # Chọn cha mẹ để lai
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                # Tạo đột biến ở con 1
                child1.update_fitness()
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()

                # Tạo đột biến ở con 2
                child2.update_fitness()
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()

                # Thêm 2 con vừa sinh ra vào quần thể
                next_population.append(child1)
                next_population.append(child2)

            for e in range(0, Ne):
                next_population.append(elites[e])

            self.population.candidates = next_population
            self.population.update_fitness()
            # Tính toán lại độ phù hợp của quần thể
        print("Không tìm được lời giải")
        return (-2, 1)