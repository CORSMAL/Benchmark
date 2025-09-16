# CORSMAL Benchmark
[![Watch the video](http://corsmal.github.io/benchmark/resources/handover.gif)](https://corsmal.github.io/benchmark/resources/Benchmark.mp4)

[Paper](https://doi.org/10.1109/LRA.2020.2969200) | [Website](https://corsmal.github.io/benchmark.html)

## Description

The real-time estimation through vision of the physical properties of objects manipulated by humans is important to inform the control of robots for performing accurate and safe grasps of objects handed over by humans. 
However, estimating the 3D pose and dimensions of previously unseen objects using only RGB cameras is challenging due to illumination variations, reflective surfaces, transparencies, and occlusions caused both by the human and the robot. 
We present a benchmark for dynamic human-to-robot handovers that do not rely on a motion capture system, markers, or prior knowledge of specific objects. 
To facilitate comparisons, the benchmark focuses on cups with different levels of transparencies and with an unknown amount of an unknown filling. 
The performance scores assess the overall system as well as its components in order to help isolate modules of the pipeline that need improvements. 
In addition to the task description and the performance scores, we also present and distribute as open source a baseline implementation for the overall pipeline to enable comparisons and facilitate progress. 

## Baseline
This repository contains the official baseline for setup 1 (S1) and setup 2 (S2).

## Reference
```
Benchmark for Human-to-Robot Handovers of Unseen Containers with Unknown Filling
R. Sanchez-Matilla, K. Chatzilygeroudis, A. Modas, N. Ferreira Duarte, A. Xompero, P. Frossard, A. Billard, and A. Cavallaro
IEEE Robotics and Automation Letters (RA-L), vol.5, no. 2, Apr. 2020
```

BibTeX
```
@Article{SanchezMatilla2020RAL_Benchmark,
        title={Benchmark for Human-to-Robot Handovers of Unseen Containers with Unknown Filling},
        author={Ricardo Sanchez-Matilla, Konstantinos Chatzilygeroudis, Apostolos Modas, Nuno Ferreira Duarte, Alessio Xompero, Pascal Frossard, Aude Billard, and Andrea Cavallaro},
        journal={IEEE Robotics and Automation Letters},
        volume={5},
        number={2},
        pages={1642--1649},
        month={Apr},
        year={2020},
        issn={2377-3766},
        doi={10.1109/LRA.2020.2969200}
      }
```
