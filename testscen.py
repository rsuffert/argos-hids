"""
Example execution of LIDS Framework
"""
import os
import sys
import datetime
from pprint import pprint

# Add LID-DS to the Python path (local directory)
LID_DS_PATH = os.path.join(os.path.dirname(__file__), "LID-DS")
if os.path.exists(LID_DS_PATH):
    sys.path.insert(0, LID_DS_PATH)
    print(f"Using local LID-DS from: {LID_DS_PATH}")
else:
    print(f"LID-DS not found at: {LID_DS_PATH}")
    print("Please run: python3 setup.py")
    sys.exit(1)

from dataloader.dataloader_factory import dataloader_factory

from dataloader.direction import Direction

from algorithms.ids import IDS

from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.and_decider import AndDecider
from algorithms.features.impl.or_decider import OrDecider
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.ngram import Ngram

from algorithms.decision_engines.stide import Stide
from algorithms.decision_engines.ae import AE

from algorithms.persistance import save_to_mongo


if __name__ == '__main__':

    # getting the LID-DS base path from argument or environment variable
    if len(sys.argv) > 1:
        LID_DS_BASE_PATH = sys.argv[1]
    else:
        try:
            LID_DS_BASE_PATH = os.environ['LID_DS_BASE']
        except KeyError as exc:
            # Try to find LID-DS datasets in common locations
            possible_paths = [
                os.path.dirname(__file__),  # Current directory (where CVE folder should be)
                os.path.join(os.path.dirname(__file__), "datasets"),
                os.path.join(os.path.expanduser("~"), "LID-DS-datasets"),
                os.path.join(os.path.expanduser("~"), "Desktop", "LID-DS"),
                os.path.join(os.path.expanduser("~"), "Documents", "LID-DS"),
                os.path.join("/data", "LID-DS"),
                os.path.join("/tmp", "LID-DS")
            ]
            
            LID_DS_BASE_PATH = None
            for path in possible_paths:
                # Check if CVE scenario exists in this path
                cve_path = os.path.join(path, "SCENARIOS", "CVE-2014-0160")
                if os.path.exists(cve_path):
                    LID_DS_BASE_PATH = path
                    print(f"Found LID-DS datasets at: {LID_DS_BASE_PATH}")
                    break
            
            if LID_DS_BASE_PATH is None:
                print("LID-DS datasets (CVE folder) not found in common locations:")
                for path in possible_paths:
                    expected_path = os.path.join(path, "CVE", "CVE-2014-0160")
                    print(f"  - {expected_path}")
                print("\nPlease either:")
                print("1. Set the LID_DS_BASE environment variable: export LID_DS_BASE=/path/to/lid-ds-datasets")
                print("2. Pass the path as an argument: python testscen.py /path/to/lid-ds-datasets")
                print("3. Make sure the 'CVE/CVE-2014-0160' folder exists in the current directory")
                sys.exit(1)

    LID_DS_VERSION = "SCENARIOS"  # Updated to use local CVE folder
    SCENARIO_NAME = "CVE-2014-0160"
    #scenario_name = "CVE-2014-0160"
    #scenario_name = "Bruteforce_CWE-307"
    #scenario_name = "CVE-2012-2122"

    scenario_path = f"{LID_DS_BASE_PATH}/{LID_DS_VERSION}/{SCENARIO_NAME}"
    print(f"Scenario path: {scenario_path}")
    # just load < closing system calls for this example
    dataloader = dataloader_factory(scenario_path,direction=Direction.BOTH)

    ### features (for more information see Paper:
    # https://dbs.uni-leipzig.de/file/EISA2021_Improving%20Host-based%20Intrusion%20Detection%20Using%20Thread%20Information.pdf
    THREAD_AWARE = True
    WINDOW_LENGTH = 1000
    NGRAM_LENGTH = 5

    ### building blocks
    # first: map each systemcall to an integer
    syscall_name = SyscallName()
    int_embedding = IntEmbedding(syscall_name)
    one_hot_encoding = OneHotEncoding(syscall_name)
    # now build ngrams from these integers
    ngram = Ngram([int_embedding], THREAD_AWARE, NGRAM_LENGTH)
    ngram_ae = Ngram([one_hot_encoding], THREAD_AWARE, NGRAM_LENGTH)
    # finally calculate the STIDE algorithm using these ngrams
    stide = Stide(ngram)
    ae = AE(ngram_ae)
    # build stream sum of stide results
    stream_sum = StreamSum(stide, False, 500, False)
    # decider threshold
    decider_1 = MaxScoreThreshold(ae)
    decider_2 = MaxScoreThreshold(stream_sum)
    combination_decider = AndDecider([decider_1, decider_2])
    ### the IDS
    ids = IDS(data_loader=dataloader,
              resulting_building_block=combination_decider,
              create_alarms=True,
              plot_switch=False)

    print("at evaluation:")
    # detection
    # normal / seriell
    # results = ids.detect().get_results()
    # parallel / map-reduce
    results = ids.detect().get_results()

    # to get alarms:
    # print(performance.alarms.alarm_list)

    ### print results
    pprint(results)

    # enrich results with configuration and save to mongoDB
    results['config'] = ids.get_config_tree_links()
    results['scenario'] = SCENARIO_NAME
    results['dataset'] = LID_DS_VERSION
    results['direction'] = dataloader.get_direction_string()
    results['date'] = str(datetime.datetime.now().date())

    # save_to_mongo(results)
