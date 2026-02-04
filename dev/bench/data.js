window.BENCHMARK_DATA = {
  "lastUpdate": 1770238894931,
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
          "id": "282f2c74a4c3eba17e01b60ba1147abac192d76f",
          "message": "Merge pull request #61 from steventango/perf\n\nfeat: remove teleport functionality",
          "timestamp": "2026-01-14T22:01:46-07:00",
          "tree_id": "5ceecdc72f4050db4ea6f6579865ca59055f4a37",
          "url": "https://github.com/steventango/continual-foragax/commit/282f2c74a4c3eba17e01b60ba1147abac192d76f"
        },
        "date": 1768453382368,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6988.7807836359225,
            "unit": "iter/sec",
            "range": "stddev: 0.000010448728519975473",
            "extra": "mean: 143.08647401582238 usec\nrounds: 1424"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 91.7666957298559,
            "unit": "iter/sec",
            "range": "stddev: 0.001944861561555327",
            "extra": "mean: 10.897199599992291 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2204.9789269413964,
            "unit": "iter/sec",
            "range": "stddev: 0.00002448867039615401",
            "extra": "mean: 453.5190734848133 usec\nrounds: 1633"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 67.35524049301586,
            "unit": "iter/sec",
            "range": "stddev: 0.0005952395881843813",
            "extra": "mean: 14.846654732139086 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.44233960751698725,
            "unit": "iter/sec",
            "range": "stddev: 0.04663072696629913",
            "extra": "mean: 2.2607064413999978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7795253830364022,
            "unit": "iter/sec",
            "range": "stddev: 0.027152522997142752",
            "extra": "mean: 1.2828318638000042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 79.77545577743817,
            "unit": "iter/sec",
            "range": "stddev: 0.000878360240822412",
            "extra": "mean: 12.535183788731379 msec\nrounds: 71"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 77.2166044034175,
            "unit": "iter/sec",
            "range": "stddev: 0.0011075643611732427",
            "extra": "mean: 12.950582426229316 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 676.7196034162508,
            "unit": "iter/sec",
            "range": "stddev: 0.00009647907187920156",
            "extra": "mean: 1.4777169080838626 msec\nrounds: 631"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1243.5592495862004,
            "unit": "iter/sec",
            "range": "stddev: 0.000032074159472827235",
            "extra": "mean: 804.1434297020863 usec\nrounds: 1010"
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
          "id": "45956278a0a8b1de624ceff6a5dc12ebfa2d181f",
          "message": "Merge pull request #62 from steventango/perf\n\nPerf",
          "timestamp": "2026-01-15T09:56:43-07:00",
          "tree_id": "a55e075f5805ecee58ffc2b8da27db88ab04dc0f",
          "url": "https://github.com/steventango/continual-foragax/commit/45956278a0a8b1de624ceff6a5dc12ebfa2d181f"
        },
        "date": 1768496371955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6942.751076663724,
            "unit": "iter/sec",
            "range": "stddev: 0.000011011722331059094",
            "extra": "mean: 144.0351222386819 usec\nrounds: 1358"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 80.4071134372361,
            "unit": "iter/sec",
            "range": "stddev: 0.0017678070251659348",
            "extra": "mean: 12.436710599996559 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2466.093086314772,
            "unit": "iter/sec",
            "range": "stddev: 0.00002742885294355114",
            "extra": "mean: 405.4996972942165 usec\nrounds: 1774"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 68.50572229544035,
            "unit": "iter/sec",
            "range": "stddev: 0.0007898679933231295",
            "extra": "mean: 14.597320727272425 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.39507167544532384,
            "unit": "iter/sec",
            "range": "stddev: 0.031936220614222556",
            "extra": "mean: 2.5311862686000013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7131423692915353,
            "unit": "iter/sec",
            "range": "stddev: 0.019136400448543842",
            "extra": "mean: 1.4022445489999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 72.47821381654101,
            "unit": "iter/sec",
            "range": "stddev: 0.0007324455498354702",
            "extra": "mean: 13.79724950909013 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 77.79962042141602,
            "unit": "iter/sec",
            "range": "stddev: 0.0011739074582158473",
            "extra": "mean: 12.853533148147964 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1825.614842406603,
            "unit": "iter/sec",
            "range": "stddev: 0.00003387588462185673",
            "extra": "mean: 547.7606649394663 usec\nrounds: 1158"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1200.2142124098964,
            "unit": "iter/sec",
            "range": "stddev: 0.00006487958313675278",
            "extra": "mean: 833.184601265562 usec\nrounds: 790"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 1603.785652862029,
            "unit": "iter/sec",
            "range": "stddev: 0.000022654067338265325",
            "extra": "mean: 623.5247199121991 usec\nrounds: 1371"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.18759935376445494,
            "unit": "iter/sec",
            "range": "stddev: 0.006287724938798937",
            "extra": "mean: 5.330508767399993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.2905920982545356,
            "unit": "iter/sec",
            "range": "stddev: 0.0037158180482376782",
            "extra": "mean: 3.441249799999997 sec\nrounds: 5"
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
          "id": "92128f3d92e270635f45524973b775389c4d635b",
          "message": "Merge pull request #64 from steventango/fix-python-support\n\nfix: jax python 3.8 compatibility",
          "timestamp": "2026-01-21T18:30:15-07:00",
          "tree_id": "4c74cb1f76194561a5c77acf8713c3f380b74379",
          "url": "https://github.com/steventango/continual-foragax/commit/92128f3d92e270635f45524973b775389c4d635b"
        },
        "date": 1769045571931,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 7031.295834315408,
            "unit": "iter/sec",
            "range": "stddev: 0.000011155055669195874",
            "extra": "mean: 142.22129513021173 usec\nrounds: 2382"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 91.51190310891938,
            "unit": "iter/sec",
            "range": "stddev: 0.0017646689198554087",
            "extra": "mean: 10.927540199986652 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2457.583273032527,
            "unit": "iter/sec",
            "range": "stddev: 0.000019643644370650803",
            "extra": "mean: 406.90381114372303 usec\nrounds: 1705"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 88.08924415668655,
            "unit": "iter/sec",
            "range": "stddev: 0.0010251341239435109",
            "extra": "mean: 11.352123741932386 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.46672127753666637,
            "unit": "iter/sec",
            "range": "stddev: 0.0498621097656371",
            "extra": "mean: 2.1426064079999834 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7744298559347964,
            "unit": "iter/sec",
            "range": "stddev: 0.021499013819647174",
            "extra": "mean: 1.2912725307999948 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 72.1028496547337,
            "unit": "iter/sec",
            "range": "stddev: 0.0007286766624579847",
            "extra": "mean: 13.869077363634378 msec\nrounds: 66"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 73.80248956007695,
            "unit": "iter/sec",
            "range": "stddev: 0.0011844491025485057",
            "extra": "mean: 13.549678418178246 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1804.7099610685493,
            "unit": "iter/sec",
            "range": "stddev: 0.000033296375103217755",
            "extra": "mean: 554.1056577356678 usec\nrounds: 1493"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1279.578182354968,
            "unit": "iter/sec",
            "range": "stddev: 0.00005762574734050858",
            "extra": "mean: 781.5075419303998 usec\nrounds: 1109"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 1593.0845810069982,
            "unit": "iter/sec",
            "range": "stddev: 0.00002149718614943877",
            "extra": "mean: 627.713061768443 usec\nrounds: 1538"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.1873885531994418,
            "unit": "iter/sec",
            "range": "stddev: 0.020365938206675756",
            "extra": "mean: 5.336505261000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.29031040663184343,
            "unit": "iter/sec",
            "range": "stddev: 0.004839482836225351",
            "extra": "mean: 3.444588885399992 sec\nrounds: 5"
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
          "id": "62f8643fc71fdc90a54e001faac77ae91f7d7030",
          "message": "Merge pull request #68 from steventango/fix-respawn\n\nFix respawn",
          "timestamp": "2026-02-04T13:58:49-07:00",
          "tree_id": "3d9d368a50e654a24cb4e3126831cb1a36713335",
          "url": "https://github.com/steventango/continual-foragax/commit/62f8643fc71fdc90a54e001faac77ae91f7d7030"
        },
        "date": 1770238893995,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 7226.181705320764,
            "unit": "iter/sec",
            "range": "stddev: 0.000009861417767596507",
            "extra": "mean: 138.38567043832882 usec\nrounds: 1414"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 86.59568429968921,
            "unit": "iter/sec",
            "range": "stddev: 0.0021273077379940887",
            "extra": "mean: 11.547919600002388 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2218.433621494235,
            "unit": "iter/sec",
            "range": "stddev: 0.000021591013779809373",
            "extra": "mean: 450.7685018434069 usec\nrounds: 1628"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 65.61078740488757,
            "unit": "iter/sec",
            "range": "stddev: 0.0008677586777642844",
            "extra": "mean: 15.241396111114291 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.43768587171993595,
            "unit": "iter/sec",
            "range": "stddev: 0.009006300648156417",
            "extra": "mean: 2.2847436132000043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.6976274034821668,
            "unit": "iter/sec",
            "range": "stddev: 0.02224064573888643",
            "extra": "mean: 1.433429929800002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 72.06645482810215,
            "unit": "iter/sec",
            "range": "stddev: 0.0007614629108174451",
            "extra": "mean: 13.876081491524298 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 69.06388722463771,
            "unit": "iter/sec",
            "range": "stddev: 0.001011464357527225",
            "extra": "mean: 14.479347169489497 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1775.966807256934,
            "unit": "iter/sec",
            "range": "stddev: 0.000060531124141077796",
            "extra": "mean: 563.0735866874383 usec\nrounds: 1292"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1194.8976347251755,
            "unit": "iter/sec",
            "range": "stddev: 0.00010062647244518713",
            "extra": "mean: 836.8917729341713 usec\nrounds: 1101"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 1747.2288362660966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254518103624041",
            "extra": "mean: 572.3348763731734 usec\nrounds: 1456"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.18508800592046637,
            "unit": "iter/sec",
            "range": "stddev: 0.006587859516207703",
            "extra": "mean: 5.40283523519999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.28116630555619376,
            "unit": "iter/sec",
            "range": "stddev: 0.006578828330046088",
            "extra": "mean: 3.5566139336000218 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}