import time


class Clock:
    def __init__(self):
        self.start_time = Clock._get_now()

    @staticmethod
    def _get_now() -> float:
        return time.perf_counter()

    def start(self):
        self.start_time = Clock._get_now()

    def stop(self):
        now = Clock._get_now()
        duration = now - self.start_time
        self.start_time = now
        return duration

    def get_duration(self):
        return Clock._get_now() - self.start_time

    def stop_and_print(self, format_str: str):
        duration = self.stop()
        print(format_str.format(duration))
