import time


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.path, 'a') as log_file:
            log_file.write(f'[{timestamp}]\t{message}\n')
        print(message)
