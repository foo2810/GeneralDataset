# Base 

import numpy as np

# Chainのメタ情報を管理するクラス
class GeneralDatasetChainBase:
    def __init__(self, kind):
        """
        self.prev = None
        self.next = None
        """
        self.kind = kind
