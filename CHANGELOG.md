## 0.8.2 (2025-09-21)

### Fix

- add package-data configuration

## 0.8.1 (2025-09-21)

### Fix

- data packaging

## 0.8.0 (2025-09-21)

### Feat

- add temperature into info
- implement ForagerWeather

### Fix

- ensure fieldnames are stripped of whitespace in load_data function
- rename Forager to Foragax

### Refactor

- remove TODO comment

## 0.7.0 (2025-09-17)

### Feat

- add Python 3.8 to the test matrix
- support Python 3.8

### Fix

- remove version constraints

## 0.6.0 (2025-09-17)

### Feat

- support Python 3.9
- rename ForagaxTwoBiomeSmall to ForagaxTwoBiomeSmall-v1

### Fix

- add six dependency for python3.9
- lock env for python3.9

## 0.5.0 (2025-09-15)

### Feat

- remove ForagaxTwoBiomeSmall100
- ForagaxTwoBiomeSmall-v2

## 0.4.1 (2025-09-10)

### Fix

- use python types to avoid warning

## 0.4.0 (2025-09-09)

### Feat

- add ForagaxTwoBiomeSmall100 config

### Fix

- use dtype float_ instead of float32

## 0.3.2 (2025-09-02)

### Fix

- use dtype int_ instead of int32
- use new-style RNG keys

## 0.3.1 (2025-08-27)

### Fix

- use Large Morel and Large Oyster in TwoBiomeSmall environment

## 0.3.0 (2025-08-26)

### Feat

- visualize both world and aperture views
- add aperture render mode

### Fix

- observations where aperture_size > size

## 0.2.1 (2025-08-21)

### Fix

- remove key from state space

### Refactor

- remove unimplemented get_obs methods

## 0.2.0 (2025-08-21)

### Feat

- use int for color
