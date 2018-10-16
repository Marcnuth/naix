"""Provide Implementation of Android Device
"""
import os
import cv2
import time
import logging
import subprocess
import numpy as np
from itertools import chain
from pathlib import Path
from naix.interfaces.basic import BasicInterface


logger = logging.getLogger(__name__)


class AndroidAdbInterface(BasicInterface):
    """
    Android Interface via ABD
    """

    _OPERATION_FUNC_PREFIX = '_execute_'

    def __init__(self, **kwargs):
        super(AndroidAdbInterface, self).__init__(**kwargs)
        self._device_id = kwargs['device_id']
        self._id = self._device_id
        self._operations = list(map(
            lambda x: x.replace(self._OPERATION_FUNC_PREFIX, ''),
            filter(lambda x: x.startswith(self._OPERATION_FUNC_PREFIX), dir(self))
        ))

    def screenshot(self):
        proc = subprocess.Popen('adb -s {} exec-out screencap -p'.format(self._device_id), shell=True, stdout=subprocess.PIPE)
        return cv2.imdecode(np.frombuffer(proc.stdout.read(), np.uint8), cv2.IMREAD_COLOR)

    def start(self, **kwargs):
        actions = kwargs.get('actions', [])
        for action in actions:
            self.execute(action, ignore_errors=kwargs.get('ignore_errors', True))

    def exit(self, **kwargs):
        actions = kwargs.get('actions', [])
        for action in actions:
            self.execute(action, ignore_errors=kwargs.get('ignore_errors', True))

    def execute(self, action, ignore_errors=True, interval_seconds=2):
        """
        Execute action
        :param action: format: tuple(operation, *operation_arguments)
        :return: self
        """
        try:
            assert action[0] in self._operations, 'operation:{} is invalid'.format(action[0])
            if len(action) > 1:
                getattr(self, self._OPERATION_FUNC_PREFIX + action[0])(*action[1:])
            else:
                getattr(self, self._OPERATION_FUNC_PREFIX + action[0])()
            time.sleep(interval_seconds)
        except Exception as e:
            if ignore_errors:
                logger.warning('Ignore execute exception:% when execute:%s', str(e), action)
            else:
                raise e

        return self

    def is_running_page(self, package_name):
        proc = subprocess.Popen('adb shell dumpsys window windows', shell=True, stdout=subprocess.PIPE)
        result = proc.stdout.read()
        line = next((x.decode() for x in result.splitlines() if b'mCurrentFocus' in x), None)
        return line and package_name in line

    def _execute_adb_command(self, operation, context):
        os.system('adb -s {} shell {} {}'.format(self._device_id, operation, context))

    def _execute_click(self, coordinate):
        self._execute_adb_command(operation='input tap', context='{:.0f} {:.0f}'.format(*coordinate))

    def _execute_drag(self, coordinates):
        self._execute_adb_command(operation='input swipe', context='{:.0f} {:.0f} {:.0f} {:.0f}'.format(*coordinates))

    def _execute_input(self, to_input):
        self._execute_adb_command(operation='input text', context=to_input)

    def _execute_back(self):
        self._execute_adb_command(operation='input keyevent', context=str(4))

    def _execute_startapp(self, package_name):
        self._execute_adb_command(operation='am start', context='-n {}/.MainActivity'.format(package_name))

    def _execute_exitapp(self, package_name):
        self._execute_adb_command(operation='am force-stop', context=package_name)



