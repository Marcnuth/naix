from abc import ABC, abstractmethod
import uuid
import time


class BasicInterface(ABC):
    def __init__(self, **kwargs):
        self._id = str(uuid.uuid4())

    def id(self):
        return self._id

    def restart(self, **kwargs):
        """Restart the interface"""
        self.exit(**kwargs)
        self.start(**kwargs)
        return self

    def executes(self, actions, interval_seconds=2):
        for action in actions:
            self.execute(action, interval_seconds=interval_seconds)

    @abstractmethod
    def start(self, **kwargs):
        """Start the interface"""
        return self

    @abstractmethod
    def exit(self, **kwargs):
        """Exit the interface"""
        return self

    @abstractmethod
    def execute(self, action, ignore_errors=True, interval_seconds=2):
        """
        Execute an action on the interface
        :param action: action to execute
        """
        return self

    @abstractmethod
    def screenshot(self):
        pass
