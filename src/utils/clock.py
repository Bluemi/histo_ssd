import time


class Clock:
    def __init__(self):
        self.start_time = time.clock()

    def start(self):
        self.start_time = time.clock()

    def stop(self):
        now = time.clock()
        duration = now - self.start_time
        self.start_time = now
        return duration

    def stop_and_print(self, format_str: str):
        duration = self.stop()
        print(format_str.format(duration))
