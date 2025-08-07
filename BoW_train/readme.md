This folder follows the structure of [ORB\_SLAM2](https://github.com/raulmur/ORB_SLAM2) and [DBoW2](https://github.com/dorian3d/DBoW2)

It includes a BoW (Bag of Words) training program that allows you to generate a custom vocabulary compatible with the ORB-SLAM2 system.
You can use the trained vocabulary by simply replacing the original vocabulary file—no other modifications to ORB-SLAM2 are required.

## Result 
MH_01
| Type                               | Voc Size | Min | Max | Mean | Std |
| ---------------------------------- | -------- | --- | --- | ---- | --- |
| ORB\_SLAM2                         | 145.3 MB |  0.0417   |   0.0499  |   0.0449   |  0.0018   |
| ORB\_SLAM2 + CoCo Voc              | 29.5 MB  |  0.0397   |   0.0540  |   0.0462   |  0.0028   |
| ORB\_SLAM2 + Genome Voc            | 22.8 MB  |  0.0414   |   0.0610  |   0.0457   |  0.0027   |
| ORB\_SLAM2 + Hpatches Voc          | 3.8 MB   |  0.0422   |   0.0531  |   0.0455   |  0.0020   |
| ORB\_SLAM2 + HP+CoCo+Genome        | 28.6 MB  |  0.0416   |   0.0541  |   0.0457   |  0.0023   |

MH_05
| Type                               | Voc Size | Min | Max | Mean | Std |
| ---------------------------------- | -------- | --- | --- | ---- | --- |
| ORB\_SLAM2                         | 145.3 MB |  0.0453   |  0.7361   |   0.0949   |   0.1337  |
| ORB\_SLAM2 + CoCo Voc              | 29.5 MB  |  0.0479   |  0.8676   |   0.0854   |   0.1137  |
| ORB\_SLAM2 + Genome Voc            | 22.8 MB  |  0.0477   |  0.2770   |   0.0649   |   0.0320  |
| ORB\_SLAM2 + Hpatches Voc          | 3.8 MB   |  0.0466   |  0.6987   |   0.1076   |   0.1332  |
| ORB\_SLAM2 + HP+CoCo+Genome        | 28.6 MB  |  0.0450   |  0.6101   |   0.0700   |   0.0715  |
