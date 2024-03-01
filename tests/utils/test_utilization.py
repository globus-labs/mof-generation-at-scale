import json
from mofa.hpc.utilization import get_utilization


def test_utilization():
    result = json.dumps(get_utilization(), indent=2)  # Make sure it will render
    print(result)
