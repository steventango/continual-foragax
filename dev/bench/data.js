window.BENCHMARK_DATA = {
  "lastUpdate": 1768452961447,
  "repoUrl": "https://github.com/steventango/continual-foragax",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "18170455+steventango@users.noreply.github.com",
            "name": "Steven Tang",
            "username": "steventango"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "599630a4588e49545b386387fd64a57f2f3971e3",
          "message": "Merge pull request #59 from steventango/benchmark\n\nBenchmark",
          "timestamp": "2026-01-14T21:40:37-07:00",
          "tree_id": "29d2085b3d9b6244243c4e6e58438aa0c6fc0673",
          "url": "https://github.com/steventango/continual-foragax/commit/599630a4588e49545b386387fd64a57f2f3971e3"
        },
        "date": 1768452291080,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 8535.929520851252,
            "unit": "iter/sec",
            "range": "stddev: 0.0000085370536705337",
            "extra": "mean: 117.15185763392694 usec\nrounds: 1454"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 60.12514760874557,
            "unit": "iter/sec",
            "range": "stddev: 0.0016100356747376482",
            "extra": "mean: 16.6319757999986 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2350.119625999274,
            "unit": "iter/sec",
            "range": "stddev: 0.000030582545900285235",
            "extra": "mean: 425.5102544300479 usec\nrounds: 1580"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 44.59092256617991,
            "unit": "iter/sec",
            "range": "stddev: 0.0010174421786257708",
            "extra": "mean: 22.426089043478378 msec\nrounds: 46"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.3186379586653045,
            "unit": "iter/sec",
            "range": "stddev: 0.03395459765359255",
            "extra": "mean: 3.138358041800018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.6254780624216463,
            "unit": "iter/sec",
            "range": "stddev: 0.010577719802956877",
            "extra": "mean: 1.5987770956000076 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 54.29688365887983,
            "unit": "iter/sec",
            "range": "stddev: 0.0006361634692455141",
            "extra": "mean: 18.41726324999608 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 55.22147336812947,
            "unit": "iter/sec",
            "range": "stddev: 0.0007650922994034216",
            "extra": "mean: 18.108897481484806 msec\nrounds: 54"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "18170455+steventango@users.noreply.github.com",
            "name": "Steven Tang",
            "username": "steventango"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f128155ccffb44d8b5e1ec85fd8d2b6632d1336f",
          "message": "Merge pull request #60 from steventango/benchmark\n\ntest: benchmark ForagaxDiwali-v5 and ForagaxSineTwoBiome-v1",
          "timestamp": "2026-01-14T21:54:45-07:00",
          "tree_id": "a9acf0bdacf183e147fb794e630d7f277baf0726",
          "url": "https://github.com/steventango/continual-foragax/commit/f128155ccffb44d8b5e1ec85fd8d2b6632d1336f"
        },
        "date": 1768452961142,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 7066.050204354901,
            "unit": "iter/sec",
            "range": "stddev: 0.000010788988334988161",
            "extra": "mean: 141.52177964765755 usec\nrounds: 1248"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 82.91651202736601,
            "unit": "iter/sec",
            "range": "stddev: 0.0016086465043705482",
            "extra": "mean: 12.060324000000833 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2179.2264169758314,
            "unit": "iter/sec",
            "range": "stddev: 0.00002143174262509422",
            "extra": "mean: 458.8784314517101 usec\nrounds: 1488"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 66.5166618015718,
            "unit": "iter/sec",
            "range": "stddev: 0.0007846986195014103",
            "extra": "mean: 15.033827208333683 msec\nrounds: 48"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4225760403741696,
            "unit": "iter/sec",
            "range": "stddev: 0.03065686019177656",
            "extra": "mean: 2.366438000400001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7596442504541215,
            "unit": "iter/sec",
            "range": "stddev: 0.020265010669035598",
            "extra": "mean: 1.3164056720000077 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 83.64515943049707,
            "unit": "iter/sec",
            "range": "stddev: 0.0006952491091479687",
            "extra": "mean: 11.95526443859463 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 69.3236354127333,
            "unit": "iter/sec",
            "range": "stddev: 0.0007183119618465001",
            "extra": "mean: 14.42509461666693 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 687.2107918221914,
            "unit": "iter/sec",
            "range": "stddev: 0.00010603254373449844",
            "extra": "mean: 1.4551575905093463 msec\nrounds: 569"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1257.5167707071373,
            "unit": "iter/sec",
            "range": "stddev: 0.00003424806179678473",
            "extra": "mean: 795.2180227685327 usec\nrounds: 1098"
          }
        ]
      }
    ]
  }
}