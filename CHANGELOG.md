## 0.15.0 (2025-09-25)

### Feat

- create ForagaxTwoBiome-v3 with deathcaps with reward=-5

## 0.14.0 (2025-09-24)

### Feat

- add ForagaxTwoBiome-v2 that defaults nowrap to true

## 0.13.0 (2025-09-24)

### Feat

- add ForagaxWeather-v2
- ad ForagaxTwoBiomeSmall-v3 that defaults nowrap to true
- add default nowrap configuration to Foragax environment variants
- support same_color option in create_weather_objects

## 0.12.0 (2025-09-24)

### Feat

- implement nowrap vision
- add nowrap dynamics

## 0.11.0 (2025-09-22)

### Feat

- add ForagaxTwoBiome-v1

## 0.10.3 (2025-09-22)

### Fix

- rename up / down for consistency
- unflip observation in object mode for consistency

## 0.10.2 (2025-09-22)

### Fix

- JIT for true rendering modes

## 0.10.1 (2025-09-22)

### Fix

- world observation mode

## 0.10.0 (2025-09-22)

### Feat

- rendering of underlying state with color partial observability

### Fix

- repeat for ForagerWeather should be 500
- correct formatting of import statements

### Refactor

- extract out border and color conversion logic

## 0.9.0 (2025-09-22)

### Feat

- add color based partial observability

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
