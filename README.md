### Extended Kalman Filter SLAM (Feature-Based Implementation)

This repository contains the implementation and results of **Feature EKF SLAM**, developed as part of the *Hands-on Localization* laboratory series.
The project focuses on extending a **map-based EKF localization** system into a **feature-based SLAM** system capable of **online map building** and **state estimation**.

---

## Overview

In this project, a **feature-based EKF SLAM (FEKFSLAM)** system was implemented and tested using a **3DOF differential-drive robot** operating in a 2D environment with **Cartesian features**.

The lab is divided into three main parts:

1. **Localization without feature measurements** – baseline EKF localization
2. **SLAM with a priori-known features** – inclusion of feature updates
3. **Full SLAM** – dynamic map growth and online feature mapping

Each stage incrementally improves the estimation accuracy and extends the SLAM functionality.

---

## System Description

### Core Concepts

* **State Vector Augmentation:** robot pose + feature positions
* **Covariance Matrix Expansion:** includes robot–feature correlations
* **Prediction Step:** propagates motion uncertainty
* **Update Step:** refines state using feature observations
* **Feature Addition:** integrates newly detected features online

### Implemented Models

* **Motion Model:** differential-drive kinematics
* **Observation Model:** 2D Cartesian feature measurements
* **Jacobian Computations:** for both motion and observation models
* **Inverse Observation Model (g):** computes feature global position from robot pose and local measurement

---

## Simulation Results

* ✅ Part I: *Prediction only* — uncertainty grows over time; robot drifts from ground truth.
* ✅ Part II: *Prediction + Update* — improved accuracy; reduced estimation error.
* ✅ Part III: *Full SLAM* — robot and feature estimates align closely with ground truth; uncertainty ellipses stabilize.

> The full SLAM implementation shows smooth trajectory tracking and accurate feature mapping with significantly reduced covariance growth.

---

## Known Issues

* Occasional `AttributeError` during the **boxplus** operation on the output of `o2s()`, causing the simulation to terminate unexpectedly.

---

## Conclusion

This project provided hands-on experience in developing a **feature-based EKF SLAM system**, highlighting the relationship between localization and mapping.
By incrementally building the system, a deeper understanding of **state augmentation**, **uncertainty propagation**, and **feature correlation** in SLAM was achieved.

