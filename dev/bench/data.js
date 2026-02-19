window.BENCHMARK_DATA = {
  "lastUpdate": 1771538711461,
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
          "id": "a0dae902feb9dafd5f040c17452d8b38d9332783",
          "message": "Merge pull request #70 from steventango/new-env\n\nfeat: ForagaxBig-v1",
          "timestamp": "2026-02-04T14:40:51-07:00",
          "tree_id": "ee3e73c178642526b2bf3445a11a2d6e56951b64",
          "url": "https://github.com/steventango/continual-foragax/commit/a0dae902feb9dafd5f040c17452d8b38d9332783"
        },
        "date": 1770241411192,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 7063.743535948674,
            "unit": "iter/sec",
            "range": "stddev: 0.000010696178547560169",
            "extra": "mean: 141.567993643996 usec\nrounds: 1416"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 88.66972098387214,
            "unit": "iter/sec",
            "range": "stddev: 0.0025225730869885715",
            "extra": "mean: 11.27780700000045 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2267.280030036858,
            "unit": "iter/sec",
            "range": "stddev: 0.00002054620371419579",
            "extra": "mean: 441.05711987581145 usec\nrounds: 1610"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 75.38104088567817,
            "unit": "iter/sec",
            "range": "stddev: 0.0010570921230426433",
            "extra": "mean: 13.265935150943138 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4622201803117799,
            "unit": "iter/sec",
            "range": "stddev: 0.007336523191938564",
            "extra": "mean: 2.1634710958 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7443563307559753,
            "unit": "iter/sec",
            "range": "stddev: 0.011579291898880168",
            "extra": "mean: 1.343442594199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 76.38017846167723,
            "unit": "iter/sec",
            "range": "stddev: 0.0008404068727933051",
            "extra": "mean: 13.092401983608053 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 75.08708865052861,
            "unit": "iter/sec",
            "range": "stddev: 0.0010421893512883766",
            "extra": "mean: 13.317868863636651 msec\nrounds: 66"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1867.7938819398055,
            "unit": "iter/sec",
            "range": "stddev: 0.00002497601943688118",
            "extra": "mean: 535.3909816651961 usec\nrounds: 1309"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1296.5221190015984,
            "unit": "iter/sec",
            "range": "stddev: 0.00004153869273772017",
            "extra": "mean: 771.2942072828355 usec\nrounds: 1071"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 1773.1501266973032,
            "unit": "iter/sec",
            "range": "stddev: 0.00001103173052333583",
            "extra": "mean: 563.9680391093649 usec\nrounds: 1662"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.18583335078277516,
            "unit": "iter/sec",
            "range": "stddev: 0.0018962682273489496",
            "extra": "mean: 5.381165414000003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.28350520402753543,
            "unit": "iter/sec",
            "range": "stddev: 0.008319697756645268",
            "extra": "mean: 3.5272721128 sec\nrounds: 5"
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
          "id": "7d1d6dc3a26dcba3815390c8a6339e928d5d487e",
          "message": "Merge pull request #76 from steventango/new-env\n\nForagaxBig-v2: reward_centering, adjust render scale, fix benchmark ci",
          "timestamp": "2026-02-06T09:15:30-07:00",
          "tree_id": "b42d61e5c13496c424a83ae94b9fcf5a22b0c8cb",
          "url": "https://github.com/steventango/continual-foragax/commit/7d1d6dc3a26dcba3815390c8a6339e928d5d487e"
        },
        "date": 1770394712929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6890.830520108592,
            "unit": "iter/sec",
            "range": "stddev: 0.000010638795757694089",
            "extra": "mean: 145.12038818569596 usec\nrounds: 2370"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 85.95129666043405,
            "unit": "iter/sec",
            "range": "stddev: 0.001879395172675487",
            "extra": "mean: 11.634495799995648 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2218.682108362905,
            "unit": "iter/sec",
            "range": "stddev: 0.000020273809202908742",
            "extra": "mean: 450.71801689421306 usec\nrounds: 1539"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 76.08116960794146,
            "unit": "iter/sec",
            "range": "stddev: 0.0010653397748540546",
            "extra": "mean: 13.14385681967248 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4341242644580925,
            "unit": "iter/sec",
            "range": "stddev: 0.022693165621802756",
            "extra": "mean: 2.303487922400001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7220042036829121,
            "unit": "iter/sec",
            "range": "stddev: 0.0056642737011497295",
            "extra": "mean: 1.3850334871999963 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 78.26514533012922,
            "unit": "iter/sec",
            "range": "stddev: 0.0011704416189636181",
            "extra": "mean: 12.777079704917337 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 72.28138807062932,
            "unit": "iter/sec",
            "range": "stddev: 0.0006991241455976246",
            "extra": "mean: 13.834820092592247 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1869.6586318741918,
            "unit": "iter/sec",
            "range": "stddev: 0.000022995446556643713",
            "extra": "mean: 534.8569963264232 usec\nrounds: 1361"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1255.2483604048368,
            "unit": "iter/sec",
            "range": "stddev: 0.00004277269394507023",
            "extra": "mean: 796.6550935605163 usec\nrounds: 823"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 6405.428522225626,
            "unit": "iter/sec",
            "range": "stddev: 0.000010586789557972957",
            "extra": "mean: 156.11758003858586 usec\nrounds: 4679"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.1640464077604878,
            "unit": "iter/sec",
            "range": "stddev: 0.002394487190293973",
            "extra": "mean: 6.095836011600005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.2846939084043446,
            "unit": "iter/sec",
            "range": "stddev: 0.0016824312412210948",
            "extra": "mean: 3.5125444222000057 sec\nrounds: 5"
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
          "id": "4ff38dfafc6c9a831baf96f0a64a9f1fd491a6e5",
          "message": "Merge pull request #77 from steventango/new-env\n\nfix: reward center with global mean reward",
          "timestamp": "2026-02-06T09:58:58-07:00",
          "tree_id": "c1daf9a19f57c3deb79793a3d93607b42b8cadb3",
          "url": "https://github.com/steventango/continual-foragax/commit/4ff38dfafc6c9a831baf96f0a64a9f1fd491a6e5"
        },
        "date": 1770397315479,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6963.094501736252,
            "unit": "iter/sec",
            "range": "stddev: 0.000012866692353372145",
            "extra": "mean: 143.61430822899922 usec\nrounds: 1288"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 85.45844705213946,
            "unit": "iter/sec",
            "range": "stddev: 0.001531434401642848",
            "extra": "mean: 11.701593400005095 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2214.306993772773,
            "unit": "iter/sec",
            "range": "stddev: 0.000024557526727113324",
            "extra": "mean: 451.6085632264492 usec\nrounds: 1463"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 63.36201296958963,
            "unit": "iter/sec",
            "range": "stddev: 0.0011955551191171472",
            "extra": "mean: 15.78232687272651 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.37926219250429666,
            "unit": "iter/sec",
            "range": "stddev: 0.03123019646800546",
            "extra": "mean: 2.636698357399996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.6567268636471939,
            "unit": "iter/sec",
            "range": "stddev: 0.012264947729680977",
            "extra": "mean: 1.5227030525999907 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 70.66498341239877,
            "unit": "iter/sec",
            "range": "stddev: 0.0010327967514164598",
            "extra": "mean: 14.151280474574364 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 69.39012887626089,
            "unit": "iter/sec",
            "range": "stddev: 0.0010937934701650074",
            "extra": "mean: 14.4112716923071 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1872.5513357371176,
            "unit": "iter/sec",
            "range": "stddev: 0.000027427778107835743",
            "extra": "mean: 534.0307530775153 usec\nrounds: 1300"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1251.8488796513757,
            "unit": "iter/sec",
            "range": "stddev: 0.00003480580297201288",
            "extra": "mean: 798.8184646364724 usec\nrounds: 919"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 6398.575448571283,
            "unit": "iter/sec",
            "range": "stddev: 0.000014982409204508343",
            "extra": "mean: 156.28478683068226 usec\nrounds: 4799"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.1581566892126336,
            "unit": "iter/sec",
            "range": "stddev: 0.04300919521611674",
            "extra": "mean: 6.3228435356000094 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.2780276441717775,
            "unit": "iter/sec",
            "range": "stddev: 0.048112301163974476",
            "extra": "mean: 3.5967646417999957 sec\nrounds: 5"
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
          "id": "599f230010eccd8bccf5791faf1c380c362cd8fc",
          "message": "Merge pull request #78 from steventango/new-env\n\nfeat: biome regret and rank metrics",
          "timestamp": "2026-02-09T14:21:11-07:00",
          "tree_id": "925b6945567b9e31a20ece4c107865dd4febea0f",
          "url": "https://github.com/steventango/continual-foragax/commit/599f230010eccd8bccf5791faf1c380c362cd8fc"
        },
        "date": 1770672241397,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6881.616172468751,
            "unit": "iter/sec",
            "range": "stddev: 0.000009619156840839882",
            "extra": "mean: 145.3147015087379 usec\nrounds: 1856"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 80.73184124523037,
            "unit": "iter/sec",
            "range": "stddev: 0.0013000032675997062",
            "extra": "mean: 12.38668640000924 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2224.637389191637,
            "unit": "iter/sec",
            "range": "stddev: 0.000020726371422645283",
            "extra": "mean: 449.51145964662965 usec\nrounds: 1586"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 69.89697143027428,
            "unit": "iter/sec",
            "range": "stddev: 0.0009984902751613693",
            "extra": "mean: 14.306771517240202 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4475879108377992,
            "unit": "iter/sec",
            "range": "stddev: 0.014074059807850382",
            "extra": "mean: 2.234197965999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7263485203589151,
            "unit": "iter/sec",
            "range": "stddev: 0.018364259023222886",
            "extra": "mean: 1.376749552000001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 77.03572353975662,
            "unit": "iter/sec",
            "range": "stddev: 0.0010815680178997986",
            "extra": "mean: 12.98099055932044 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 75.75962203261179,
            "unit": "iter/sec",
            "range": "stddev: 0.0008773008846821926",
            "extra": "mean: 13.199643466667984 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1863.346514011329,
            "unit": "iter/sec",
            "range": "stddev: 0.000023717236484609466",
            "extra": "mean: 536.6688334566633 usec\nrounds: 1351"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1223.6848841845288,
            "unit": "iter/sec",
            "range": "stddev: 0.00004370839241752499",
            "extra": "mean: 817.2038511911555 usec\nrounds: 1008"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 6486.75299592944,
            "unit": "iter/sec",
            "range": "stddev: 0.000008892652340434765",
            "extra": "mean: 154.16033270073933 usec\nrounds: 5275"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.16360002901463494,
            "unit": "iter/sec",
            "range": "stddev: 0.0061600899454800255",
            "extra": "mean: 6.112468353600013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.283486570294204,
            "unit": "iter/sec",
            "range": "stddev: 0.0018495327122596756",
            "extra": "mean: 3.527503962400033 sec\nrounds: 5"
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
          "id": "66a46b484bb43ddc6f6c3d908d2ff53d3c099e7c",
          "message": "Merge pull request #79 from steventango/new-env\n\nfix: negative regret",
          "timestamp": "2026-02-09T15:12:20-07:00",
          "tree_id": "979bf5aafbbbece81e8f661468ee0a24237e0512",
          "url": "https://github.com/steventango/continual-foragax/commit/66a46b484bb43ddc6f6c3d908d2ff53d3c099e7c"
        },
        "date": 1770675309953,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6893.27679432714,
            "unit": "iter/sec",
            "range": "stddev: 0.000011176646932810603",
            "extra": "mean: 145.06888811181287 usec\nrounds: 1716"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 97.69338875233748,
            "unit": "iter/sec",
            "range": "stddev: 0.0026186106038487704",
            "extra": "mean: 10.236107199997946 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2206.560642631722,
            "unit": "iter/sec",
            "range": "stddev: 0.000023021262605069433",
            "extra": "mean: 453.19398011527994 usec\nrounds: 1559"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 67.42919606462749,
            "unit": "iter/sec",
            "range": "stddev: 0.0007024543262973109",
            "extra": "mean: 14.830371090907718 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4216049225242964,
            "unit": "iter/sec",
            "range": "stddev: 0.059876697478158956",
            "extra": "mean: 2.371888814800002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7114749513821331,
            "unit": "iter/sec",
            "range": "stddev: 0.041555775239164684",
            "extra": "mean: 1.4055308596000031 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 70.08734830860362,
            "unit": "iter/sec",
            "range": "stddev: 0.0010271892444228346",
            "extra": "mean: 14.267910316664729 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 70.50907445995466,
            "unit": "iter/sec",
            "range": "stddev: 0.001346697074905595",
            "extra": "mean: 14.182571642859187 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1807.6367670192383,
            "unit": "iter/sec",
            "range": "stddev: 0.00003105356359631162",
            "extra": "mean: 553.2084864864653 usec\nrounds: 1295"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1185.6711903837142,
            "unit": "iter/sec",
            "range": "stddev: 0.00006951891575565233",
            "extra": "mean: 843.404147887218 usec\nrounds: 852"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 6508.827950956939,
            "unit": "iter/sec",
            "range": "stddev: 0.000010111788015623036",
            "extra": "mean: 153.63749165515708 usec\nrounds: 5093"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.16300856183046888,
            "unit": "iter/sec",
            "range": "stddev: 0.004645507514808127",
            "extra": "mean: 6.134647093200011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.28145574359797265,
            "unit": "iter/sec",
            "range": "stddev: 0.004927998303809627",
            "extra": "mean: 3.5529564513999956 sec\nrounds: 5"
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
          "id": "46754b57ea19c33680478195734f101b80d29616",
          "message": "Merge pull request #81 from steventango/new-env\n\nfeat: ForagaxBig-v3",
          "timestamp": "2026-02-19T14:53:04-07:00",
          "tree_id": "cba9703e6b8fc72ded0dd3e55d74e455368b929f",
          "url": "https://github.com/steventango/continual-foragax/commit/46754b57ea19c33680478195734f101b80d29616"
        },
        "date": 1771538156035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6910.961548199132,
            "unit": "iter/sec",
            "range": "stddev: 0.000014570609923463046",
            "extra": "mean: 144.69766515494234 usec\nrounds: 1323"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 87.72168981588106,
            "unit": "iter/sec",
            "range": "stddev: 0.0016750600030408984",
            "extra": "mean: 11.399689200001717 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2142.0647308323882,
            "unit": "iter/sec",
            "range": "stddev: 0.000025466965073797603",
            "extra": "mean: 466.83930023506264 usec\nrounds: 1702"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 70.87777415322918,
            "unit": "iter/sec",
            "range": "stddev: 0.0010734816146260142",
            "extra": "mean: 14.10879520338944 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.4468566392223375,
            "unit": "iter/sec",
            "range": "stddev: 0.054806173911702304",
            "extra": "mean: 2.237854184599999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7386228542747206,
            "unit": "iter/sec",
            "range": "stddev: 0.007992691619774",
            "extra": "mean: 1.3538709155999982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 68.41228297152041,
            "unit": "iter/sec",
            "range": "stddev: 0.001013870588826533",
            "extra": "mean: 14.617258137932534 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 76.37645337400657,
            "unit": "iter/sec",
            "range": "stddev: 0.0007471656977118773",
            "extra": "mean: 13.093040535714806 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1782.391055692718,
            "unit": "iter/sec",
            "range": "stddev: 0.0000262616759836325",
            "extra": "mean: 561.0441080290063 usec\nrounds: 1370"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1213.5113791568535,
            "unit": "iter/sec",
            "range": "stddev: 0.000035519512498754735",
            "extra": "mean: 824.0549014833292 usec\nrounds: 944"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 5944.825723038089,
            "unit": "iter/sec",
            "range": "stddev: 0.000011363492934482379",
            "extra": "mean: 168.21350979637336 usec\nrounds: 5206"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.16069841672669463,
            "unit": "iter/sec",
            "range": "stddev: 0.02059059998513243",
            "extra": "mean: 6.222836667399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.27867895606871534,
            "unit": "iter/sec",
            "range": "stddev: 0.005712674359108162",
            "extra": "mean: 3.5883584971999993 sec\nrounds: 5"
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
          "id": "123b9f06881c96f58b8f0cf75312ff445ce06081",
          "message": "Merge pull request #82 from steventango/new-env\n\nfeat: exclude walls from reward centering",
          "timestamp": "2026-02-19T15:02:14-07:00",
          "tree_id": "369ec369a577f3925600269014b56637d46db348",
          "url": "https://github.com/steventango/continual-foragax/commit/123b9f06881c96f58b8f0cf75312ff445ce06081"
        },
        "date": 1771538710560,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmark.py::test_benchmark_vision",
            "value": 6768.795947377433,
            "unit": "iter/sec",
            "range": "stddev: 0.00001098325647663029",
            "extra": "mean: 147.73676260509075 usec\nrounds: 1904"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_reset",
            "value": 87.04459118578862,
            "unit": "iter/sec",
            "range": "stddev: 0.0016233479674109493",
            "extra": "mean: 11.488364599996714 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_tiny_env",
            "value": 2129.8259412692087,
            "unit": "iter/sec",
            "range": "stddev: 0.000049456960262332644",
            "extra": "mean: 469.5219363344212 usec\nrounds: 1555"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env",
            "value": 63.08240326318158,
            "unit": "iter/sec",
            "range": "stddev: 0.0006096971623911074",
            "extra": "mean: 15.85228127450965 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_big_env",
            "value": 0.43643165248777593,
            "unit": "iter/sec",
            "range": "stddev: 0.060110374544868075",
            "extra": "mean: 2.2913095196 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_vmap_env",
            "value": 0.7020325750188486,
            "unit": "iter/sec",
            "range": "stddev: 0.016954113379276978",
            "extra": "mean: 1.4244353262000005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_color",
            "value": 68.12697280371684,
            "unit": "iter/sec",
            "range": "stddev: 0.000857866741674708",
            "extra": "mean: 14.67847401470688 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_small_env_world",
            "value": 66.91322458839058,
            "unit": "iter/sec",
            "range": "stddev: 0.0008104238097003539",
            "extra": "mean: 14.944728880597092 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_diwali_v5",
            "value": 1743.7671090105468,
            "unit": "iter/sec",
            "range": "stddev: 0.00004811712311121664",
            "extra": "mean: 573.4710758292848 usec\nrounds: 1266"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_sine_two_biome_v1",
            "value": 1178.5987633232455,
            "unit": "iter/sec",
            "range": "stddev: 0.00006258441750704205",
            "extra": "mean: 848.46516992801 usec\nrounds: 971"
          },
          {
            "name": "tests/test_benchmark.py::test_benchmark_render",
            "value": 6115.854091520961,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105294139502982",
            "extra": "mean: 163.5094600092574 usec\nrounds: 4326"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_diwali_v5_vmap",
            "value": 0.15885142110923353,
            "unit": "iter/sec",
            "range": "stddev: 0.016596285592018242",
            "extra": "mean: 6.295190770199997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_optimize.py::test_benchmark_sine_two_biome_v1_vmap",
            "value": 0.27832883959379395,
            "unit": "iter/sec",
            "range": "stddev: 0.0066614097853078405",
            "extra": "mean: 3.592872378800007 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}