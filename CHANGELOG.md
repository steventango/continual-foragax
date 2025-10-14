## 0.30.1 (2025-10-14)

### Fix

- directly apply multiplier to rewards

## 0.30.0 (2025-10-09)

### Feat

- remove ForagaxWeather-v6
- add repeat and digestion_steps parameters to make function

### Refactor

- rename digestion steps to reward delay
- rename reward_delays to reward_delay
- rename digestion_step to reward_delay

## 0.29.0 (2025-10-09)

### Feat

- temperatures in info
- ForagaxWeather-v6 with delayed rewards
- remove explicit nowrap parameter from registry
- add delayed rewards via digesetion

## 0.28.1 (2025-10-05)

### Fix

- don't add padding channel for world view

## 0.28.0 (2025-10-05)

### Feat

- observation config split into type and view #22

## 0.27.0 (2025-10-04)

### Feat

- add biome_id and object_collected_id to info #23

## 0.26.0 (2025-10-04)

### Feat

- ForagaxWeather-v5 with wrapping #26
- ForagaxWeather-v4 with random respawn #25
- create_weather_objects with random_respawn

### Fix

- ForagaxWeather-v4 same_color=True

## 0.25.0 (2025-10-02)

### Feat

- ForagaxTwoBiome-v15 with teleports every 10k
- teleport steps

### Fix

- ForagaxTwoBiome-v16 has teleport, ForagaxTwoBiome-v15 does not
- simplify furthest center logic
- go left in teleport test
- teleport to furthest biome center from current pos

## 0.24.2 (2025-10-02)

### Fix

- size for v13, v14

## 0.24.1 (2025-10-02)

### Fix

- don't add margin to ForagaxTwoBiome-v13, ForagaxTwoBiome-v14

## 0.24.0 (2025-10-02)

### Feat

- Add ForagaxTwoBiome-v13 (wrap), ForagaxTwoBiome-v14 (no random respawn, wrap), ForagaxTwoBiome-v15 (no random respawn, no wrap)

## 0.23.1 (2025-10-01)

### Fix

- don't set agent position to be empty

## 0.23.0 (2025-10-01)

### Feat

- add ForagaxTwoBiome-v12 with random object spawning and higher mushroom frequencies
- ForagaxTwoBiome-v11 with higher mushroom freqs
- add ForgaxTwoBiome-v10, with objects that respawn in random positions
- random respawn based on objects
- add random respawning within biomes

### Fix

- flipped coordinates

## 0.22.0 (2025-10-01)

### Feat

- ForagaxTwoBiome-v9 with deterministic spawning
- implement deterministic spawning

### Fix

- AttributeError: 'Biome' object has no attribute 'blocking'
- hardcoded environment name

### Refactor

- move benchmark tests to new file

## 0.21.0 (2025-09-29)

### Feat

- ForagaxTwoBiome-v8

## 0.20.1 (2025-09-26)

### Fix

- aperture tint in render_mode world

## 0.20.0 (2025-09-26)

### Feat

- add ForagaxWeather-v3

## 0.19.0 (2025-09-25)

### Feat

- ForagaxTwoBiome-v7 with dynamic margin

### Fix

- world rendering with nowrap

## 0.18.0 (2025-09-25)

### Feat

- ForagaxTwoBiome-v6 with different oyster biome distribution

## 0.17.0 (2025-09-25)

### Feat

- ForagaxTwoBiome-v5 with different morel biome distribution

## 0.16.1 (2025-09-25)

### Fix

- change to NormalRegenForagaxObject

## 0.16.0 (2025-09-25)

### Feat

- create ForagerTwoBiome-v4 with more frequently respawning morels

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
