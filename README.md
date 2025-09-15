# Topdown pose estimation with identity tracking though VitPose and SAM2.

The notebook here is an implementation of vitpose through mmpose to get pose estimation.  
The topdown strategy is used with bounding box given by the user.  

We use 
https://gitlab.com/nicololaporta/segmentation  
https://github.com/ChataingT/sam2_longer_video/tree/main  
To get the bounding box.  

## Installation
Install vitpose and modify this :

in structures/utils.py l.128 add             

```

keypoints_label=instances.keypoints_label[i].tolist()
```
so we get :
```
def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
            keypoints_label=instances.keypoints_label[i].tolist()
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results
```
