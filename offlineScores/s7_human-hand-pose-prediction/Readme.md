# Documentation for s7 measure

This measure quantifies the human hand position prediction accuracy of the system (if such predictor is available/used by the approach).

## Data

We provide six (6) trajectories of synchronized robot end-effector and human hand positions. The data are available in the `data` folder. The data have the following form:

- Each row is a different time-step (the sampling rate is 50Hz),
- The columns are sorted as follows:
    - The first three (3) columns are the robot's end-effector xyz position,
    - The last three (3) columns are the human hand's xyz position.

*An illustration of the robot's coordinate frame will be added soon...*

## Prediction score

The task of the predictor is the following: having as input the current time-step information (robot's and human's positions), predict the position of the human hand of the next time-step. This should be done per trajectory. For approaches that do not use the robot's current position as input, the target position of the human hand (last point in each trajectory) can be used as a static input to the model.