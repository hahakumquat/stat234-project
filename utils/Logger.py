class Logger():

    def __init__(self, filename):
        self.f = open(filename, 'w')

    def log(self, x):
        self.f.write(str(x) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
