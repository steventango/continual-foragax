window.BENCHMARK_DATA = {
  "lastUpdate": 1768452291868,
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
      }
    ]
  }
}