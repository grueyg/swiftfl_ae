from typing import Callable, Dict, List, Optional, Tuple, cast

class Strategy():
    def __init__(self, client_duration_info, config):
        self.client_duration_info = client_duration_info
        self.cfg = config

        pass

    def find_stable(self,):
        pass

    def find_min(self,):
        pass

    def find_stable_and_min(self,):
        pass

    def update_straggler(self,):
        pass

