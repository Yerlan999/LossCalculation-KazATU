from multiprocessing import Pool



class Looper():
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self.counter = 0
        self.result = 0

    def __iter__(self):
        return self
    def __next__(self):
        if self.result < self.end:
            self.result = self.start + self.step*self.counter
            self.counter += 1
            return round(self.result, 3)
        else:
            raise StopIteration


def sum_up_to(number):
    return number * 2

result = []

if __name__ == '__main__':
    pool = Pool()                         # Create a multiprocessing Pool
    result = pool.map(sum_up_to, Looper(0, 10000, 1))

    print("Success!")

