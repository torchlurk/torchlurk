from pathlib import Path
import pandas as pd
import os
local = Path(__file__).parent
TEST_FILENAME = local/'trash.txt'
trash =pd.read_csv(TEST_FILENAME,sep=" ",header=None)

def get_current_directory():
    print(trash)
def getname():
    print(__name__)
if __name__ == '__main__':
    get_current_directory()