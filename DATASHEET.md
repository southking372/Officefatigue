# Datasheet for OfficeFatigue

## Motivation

### For what purpose was the dataset created?

OfficeFatigue was created to support research on cognitive and physical fatigue detection using wearable sensors in naturalistic office environments. Existing fatigue datasets primarily focus on driving scenarios or controlled laboratory settings, leaving a gap in ecologically valid office fatigue sensing research.

### Who created the dataset and on behalf of which entity?

The dataset was collected by [Anonymous for review].

### Who funded the creation of the dataset?

[Anonymous for review]

## Composition

### What do the instances that comprise the dataset represent?

Each instance represents a 30-minute (OFL) or 5-minute (OFS) time window of continuous wearable sensor data collected from an office worker, along with corresponding fatigue labels.

### How many instances are there in total?

| Subset | Instances | Participants |
|--------|:---------:|:------------:|
| OFL (OfficeFatigue-Long) | 91 | 6 |
| OFS (OfficeFatigue-Short) | TBD | TBD |

### What data does each instance consist of?

Each instance contains:

1. **PPG signal**: Photoplethysmography data downsampled to 1Hz (1800 timesteps for OFL)
2. **IMU signal**: 6-axis inertial data (3-axis accelerometer + 3-axis gyroscope) at 1Hz
3. **Cognitive fatigue label**: Categorical (low / medium / high)
4. **Physical fatigue label**: Categorical (low / medium / high)
5. **Questionnaire scores**: MFI and NASA-TLX subscales
6. **Behavior context**: Activity label (static / micro_move / shift)
7. **Metadata**: Participant ID, session ID, timestamps

### Is there a label or target associated with each instance?

Yes. Each instance has two primary labels:
- **Cognitive fatigue**: low (0) / medium (1) / high (2)
- **Physical fatigue**: low (0) / medium (1) / high (2)

### Label Distribution (OFL)

| Label | Cognitive Fatigue | Physical Fatigue |
|-------|:-----------------:|:----------------:|
| Low | 46 (50.5%) | 48 (52.7%) |
| Medium | 27 (29.7%) | 26 (28.6%) |
| High | 18 (19.8%) | 17 (18.7%) |

### Is any information missing from individual instances?

No. All instances contain complete sensor data and labels.

### Are there any errors, sources of noise, or redundancies in the dataset?

- PPG signals may contain motion artifacts during high-activity periods
- Some IMU readings may be affected by sensor drift
- Quality control was performed to exclude segments with >20% missing data

## Collection Process

### How was the data associated with each instance acquired?

Data was collected using wrist-worn smartwatch devices equipped with PPG and IMU sensors. Participants wore the devices during normal office work activities.

### What mechanisms or procedures were used to collect the data?

1. Participants wore wrist-worn devices during office work sessions
2. Sensor data was continuously recorded at the native sampling rate (125Hz)
3. Fatigue assessments were collected via validated questionnaires (MFI, NASA-TLX)
4. Data was segmented into analysis units (30-min for OFL, 5-min for OFS)

### Who was involved in the data collection process?

Office workers volunteered to participate in the data collection. Research staff supervised the collection process and administered questionnaires.

### Over what timeframe was the data collected?

[Anonymous for review]

### Were any ethical review processes conducted?

Yes. The study protocol was approved by [Anonymous] Institutional Review Board. All participants provided written informed consent.

## Preprocessing

### Was any preprocessing/cleaning/labeling of the data done?

Yes:

1. **Resampling**: Raw signals (125Hz) were downsampled to 1Hz using mean aggregation
2. **Normalization**: Robust scaling using median and IQR
3. **Quality filtering**: Segments with >20% missing PPG values were excluded
4. **Label assignment**: Based on validated questionnaire thresholds

### Was the "raw" data saved in addition to the preprocessed data?

Raw data is available upon request for approved research purposes.

## Uses

### What are the intended uses of this dataset?

- Developing and evaluating fatigue detection algorithms
- Benchmarking multimodal time-series classification methods
- Studying physiological correlates of cognitive and physical fatigue
- Exploring transfer learning across fatigue dimensions

### What tasks could the dataset be used for?

- **Classification**: Predicting fatigue levels from sensor data
- **Regression**: Predicting continuous fatigue scores
- **Multi-task learning**: Joint prediction of cognitive and physical fatigue
- **Self-supervised learning**: Pre-training on unlabeled sensor data

### Is there anything about the composition of the dataset that might impact future uses?

- The dataset represents office workers in a specific cultural/geographic context
- Fatigue manifestations may differ across occupations and demographics
- The three-class labels are derived from continuous questionnaire scores

### Are there tasks for which the dataset should not be used?

- **Employee surveillance**: The dataset must not be used to monitor or evaluate employee performance
- **Clinical diagnosis**: The labels are not validated for clinical use
- **Safety-critical systems**: The dataset is not suitable for applications where fatigue misclassification could cause harm

## Distribution

### How will the dataset be distributed?

The dataset will be distributed through this GitHub repository and [additional hosting TBD].

### When will the dataset be released?

Upon acceptance of the associated publication.

### Will the dataset be distributed under a copyright or intellectual property license?

Yes. The dataset is released under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

[Anonymous for review]

### How can the owner/curator/manager of the dataset be contacted?

[Anonymous for review]

### Will the dataset be updated?

We plan to release additional subsets (OFS) and potentially expand the participant pool in future versions.

### Will older versions of the dataset continue to be supported?

Yes. All versions will be archived and accessible.
