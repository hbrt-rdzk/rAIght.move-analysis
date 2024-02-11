from abc import ABC, abstractmethod


class App(ABC):
    @abstractmethod
    def run(self, args):
        ...
