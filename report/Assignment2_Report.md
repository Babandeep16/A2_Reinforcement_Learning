# CSCN8020 � Assignment 2: Taxi-v3 Q-Learning & DQN

_Auto-generated on 2025-10-16 17:52:44_

## Environment
- **Env:** Taxi-v3 (500 states, 6 actions)
- **Rewards:** -1 per step; +20 for successful dropoff; -10 for illegal pickup/dropoff

## Algorithms
- Tabular Q-Learning
- Deep Q-Network (DQN) with one-hot state encoding

## Results

### Q-Learning Hyperparameter Sweep

### Q-Learning � alpha0.001_eps0.1

**Summary**

```
episodes: 5000
avg_steps: 185.5728
std_steps: 37.77747874276418
avg_return: -258.8598
std_return: 68.73738825384625
min_return: -605.0
max_return: 11.0

```

![returns.png](out/q_sweep/alpha0.001_eps0.1/returns.png)

![steps.png](out/q_sweep/alpha0.001_eps0.1/steps.png)

### Q-Learning � alpha0.001_eps0.2

**Summary**

```
episodes: 5000
avg_steps: 186.1128
std_steps: 37.218501798970905
avg_return: -293.4702
std_return: 75.53294322320559
min_return: -623.0
max_return: 6.0

```

![returns.png](out/q_sweep/alpha0.001_eps0.2/returns.png)

![steps.png](out/q_sweep/alpha0.001_eps0.2/steps.png)

### Q-Learning � alpha0.001_eps0.3

**Summary**

```
episodes: 5000
avg_steps: 186.5466
std_steps: 36.702749603265424
avg_return: -347.1858
std_return: 86.86828004720711
min_return: -659.0
max_return: 7.0

```

![returns.png](out/q_sweep/alpha0.001_eps0.3/returns.png)

![steps.png](out/q_sweep/alpha0.001_eps0.3/steps.png)

### Q-Learning � alpha0.01_eps0.1

**Summary**

```
episodes: 5000
avg_steps: 126.9914
std_steps: 66.4711224972168
avg_return: -160.2584
std_return: 105.68642689314461
min_return: -605.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.01_eps0.1/returns.png)

![steps.png](out/q_sweep/alpha0.01_eps0.1/steps.png)

### Q-Learning � alpha0.01_eps0.2

**Summary**

```
episodes: 5000
avg_steps: 129.481
std_steps: 66.82114365229017
avg_return: -193.1938
std_return: 120.61975891851219
min_return: -614.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.01_eps0.2/returns.png)

![steps.png](out/q_sweep/alpha0.01_eps0.2/steps.png)

### Q-Learning � alpha0.01_eps0.3

**Summary**

```
episodes: 5000
avg_steps: 133.2968
std_steps: 66.82492281896029
avg_return: -237.1712
std_return: 140.0898950337247
min_return: -659.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.01_eps0.3/returns.png)

![steps.png](out/q_sweep/alpha0.01_eps0.3/steps.png)

### Q-Learning � alpha0.1_eps0.1

**Summary**

```
episodes: 5000
avg_steps: 30.591
std_steps: 43.77987801490543
avg_return: -21.8682
std_return: 72.27760115526802
min_return: -587.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.1_eps0.1/returns.png)

![steps.png](out/q_sweep/alpha0.1_eps0.1/steps.png)

### Q-Learning � alpha0.1_eps0.2

**Summary**

```
episodes: 5000
avg_steps: 32.5326
std_steps: 43.37790379029397
avg_return: -32.3946
std_return: 80.41932660026444
min_return: -614.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.1_eps0.2/returns.png)

![steps.png](out/q_sweep/alpha0.1_eps0.2/steps.png)

### Q-Learning � alpha0.1_eps0.3

**Summary**

```
episodes: 5000
avg_steps: 36.36
std_steps: 44.89471238353131
avg_return: -48.0162
std_return: 94.9209288700864
min_return: -659.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.1_eps0.3/returns.png)

![steps.png](out/q_sweep/alpha0.1_eps0.3/steps.png)

### Q-Learning � alpha0.2_eps0.1

**Summary**

```
episodes: 5000
avg_steps: 23.4212
std_steps: 33.561838307220306
avg_return: -11.3444
std_return: 58.537261540321474
min_return: -587.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.2_eps0.1/returns.png)

![steps.png](out/q_sweep/alpha0.2_eps0.1/steps.png)

### Q-Learning � alpha0.2_eps0.2

**Summary**

```
episodes: 5000
avg_steps: 25.7456
std_steps: 34.19669692587283
avg_return: -21.1304
std_return: 67.03364972787921
min_return: -641.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.2_eps0.2/returns.png)

![steps.png](out/q_sweep/alpha0.2_eps0.2/steps.png)

### Q-Learning � alpha0.2_eps0.3

**Summary**

```
episodes: 5000
avg_steps: 28.8266
std_steps: 35.00938349128702
avg_return: -33.3884
std_return: 76.93835938879904
min_return: -659.0
max_return: 15.0

```

![returns.png](out/q_sweep/alpha0.2_eps0.3/returns.png)

![steps.png](out/q_sweep/alpha0.2_eps0.3/steps.png)


### DQN Baseline

### DQN � lr0.001_gamma0.99

**Summary**

```
episodes: 4000
avg_steps: 25.34325
std_steps: 35.26936247563173
avg_return: -37.45125
std_return: 137.17251409607357
min_return: -884.0
max_return: 15.0

```

![returns.png](out/dqn/lr0.001_gamma0.99/returns.png)

![steps.png](out/dqn/lr0.001_gamma0.99/steps.png)



