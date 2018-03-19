import os 

class Logger():

    def __init__(self, filename):
        abs_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(os.path.dirname(abs_path)):
            os.makedirs(os.path.dirname(abs_path))
        self.f = open(filename, 'w')

    def log(self, x):
        self.f.write(str(x) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
