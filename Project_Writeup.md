# Track an Object in 3D Space
## 1- Match 3D Objects
### Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.
In This section the code iterate the keypoints in each bouding box in the previous and current frames and then check, which bounding boxes have the maximum number of found key points and map these boxes together.

```C++
int prevBoxId = 0, currBoxId =0;
std::for_each(prevFrame.boundingBoxes.begin(), prevFrame.boundingBoxes.end(), [&](BoundingBox prevBb)
{
    map<int, int> currBbMatches;   //this map stores the box ids that shares the same matched key points
    //iterate through the bounding boxes in the current frame to find the mathced keypoints
    std::for_each(currFrame.boundingBoxes.begin(), currFrame.boundingBoxes.end(), [&](BoundingBox currBb)
    {
        //iterate through the keypoints matches to define the key points in each bounding box
        std::for_each(matches.begin(), matches.end(), [&] (cv::DMatch match)
        {
            cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];
            if(currBb.roi.contains(currKp.pt))
            {
                currBb.keypoints.push_back(currKp);
                cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
                if(prevBb.roi.contains(prevKp.pt))
                {
                    prevFrame.keypoints.push_back(prevKp);
                    if(currBbMatches.count(currBb.boxID) > 0)
                    {
                        currBbMatches[currBb.boxID]+=1;
                    }
                    else
                    {
                        currBbMatches.insert(make_pair(currBb.boxID, 1));
                    }
                }
            }
        });
    });
    //Map the bounding box in the previous frame with the one in the current frame depending on the max existence of the key points
    int currBbId = 0, maxCount = 0;
        std::for_each(currBbMatches.begin(), currBbMatches.end(), [&](pair<int,int> matchPair)
        {
            if(matchPair.second>maxCount)
            {
                currBbId=matchPair.first;
                maxCount=matchPair.second;
            }
        });
    bbBestMatches.insert(make_pair(prevBb.boxID, currBbId));
});
```
## 2- Compute Lidar-based TTC
### Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors.
To calculate the TTC of lidat points I used euledian clustering to filter out the outliers and get more robust estimation. Then the average of the distance are calculated in the previous and current frames to minimize the distance error in calculating the TTC.
```C++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //using clustering to get the points without outliers
    double clusterTol = 0.7;
    FilterOutliers(lidarPointsPrev, clusterTol);
    FilterOutliers(lidarPointsCurr, clusterTol);
    //Just the lane in front of the car is important to calcuate ttc
    double laneWidth = 4;//, xMinPrev = numeric_limits<double>::max(), xMinCurr = numeric_limits<double>::max();
    double halfLaneWidth = laneWidth/2;

    double avrPrev = 0, avrCurr = 0, sumPrev = 0, sumCurr = 0, prevCount = 0, currCount = 0;

    std::for_each(lidarPointsPrev.begin(), lidarPointsPrev.end(),[&](LidarPoint prevPnt)
    {
        if(abs(prevPnt.y)<=halfLaneWidth)
        {
            // xMinPrev= xMinPrev < prevPnt.x ? xMinPrev : prevPnt.x;
            sumPrev+=prevPnt.x;
            prevCount++;
        }
    });
    std::for_each(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](LidarPoint currPnt)
    {
        if(abs(currPnt.y)<=halfLaneWidth)
        {
            // xMinCurr = xMinCurr < currPnt.x ? xMinCurr : currPnt.x;
            sumCurr+=currPnt.x;
            currCount++;
        }
    });
    //calculate the ttc using the average distane of the points to minimize calculating error
    avrPrev = sumPrev/prevCount;
    avrCurr = sumCurr/currCount;
    TTC = avrCurr/(frameRate * (avrPrev - avrCurr));
    cout<<"Lidar TTC = "<<TTC<<endl;
}
```
These code snippets filter the points using KD-Tree algorithm.
```C++
void FilterOutliers(vector<LidarPoint>& lidarPoints, double tolerance)
{
    KdTree* tree = new KdTree();
    vector<LidarPoint> resultCluser;
    for(int i = 0; i < lidarPoints.size(); i++)
    {
        tree->insert(lidarPoints[i], i);
    }
    vector<vector<LidarPoint>> foundClusters = ClusterLidarPoints(lidarPoints, tree, tolerance);
    pair<int, int> clusterPair(0,0); //first = cluster index, second = cluster size
    
    for(int idx = 0; idx < foundClusters.size(); idx++)
    {
        if(foundClusters[idx].size()>clusterPair.second)
        {
            clusterPair.first=idx;
            clusterPair.second=foundClusters[idx].size();
        }
    }
    lidarPoints = foundClusters[clusterPair.first];
}
```
```C++
vector<vector<LidarPoint>> ClusterLidarPoints(const vector<LidarPoint>& lidarPoints, KdTree* tree, double tolerance)
{
    vector<vector<LidarPoint>> result;
    vector<bool> processed (lidarPoints.size(), false);
    for(int idx = 0; idx < lidarPoints.size(); idx++)
    {
        if(!processed[idx])
        {
            vector<LidarPoint> cluster;
            ClusteringHelper(idx, lidarPoints, cluster, tree, processed, tolerance);
            result.push_back(cluster);
        }
    }
    return result;
}
```
```C++
void ClusteringHelper (int idx, const vector<LidarPoint>& points, vector<LidarPoint>& resultCluster, KdTree* tree, vector<bool>& processed, double distTol)
{
    processed[idx]=true;
    resultCluster.push_back(points[idx]);
    vector<int> resultIds = tree->search(points[idx], distTol);
    for(const int id : resultIds)
    {
        if(!processed[id])
            ClusteringHelper(id, points, resultCluster, tree, processed, distTol);
    }
}
```

## 3- Associate Keypoint Correspondences with Bounding Boxes: 
### Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.
The code shrink the ROI to avoid the addind key points matching from the road.
```C++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double eDist = 0, sum = 0;
    //calculate the mean distance of all matches
    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); itr++)
    {
        sum += itr->distance; 
    }
    eDist = sum/kptMatches.size();
    double widthScaleFactor = 0.8;
    double heightScaleFactor = 0.8;
    double scaledWidth = boundingBox.roi.width*widthScaleFactor;
    double scaledHeight = boundingBox.roi.height*heightScaleFactor;
    cv::Rect scaledRoi = cv::Rect(boundingBox.roi.x + (boundingBox.roi.width - scaledWidth) / 2, boundingBox.roi.y + (boundingBox.roi.height - scaledHeight) / 2, scaledWidth, scaledHeight);
    boundingBox.roi = scaledRoi;
    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); itr++)
    {
        cv::KeyPoint prevKp = kptsPrev[itr->queryIdx];
        cv::KeyPoint currKp = kptsCurr[itr->trainIdx];
        if(std::fabs(itr->distance - eDist) < 15 && scaledRoi.contains(prevKp.pt) && scaledRoi.contains(currKp.pt))
        {
            boundingBox.kptMatches.push_back(*itr);
        }
    }
}
```

## 4- Compute Camera-based TTC: Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

## 5- Performance Evaluation 1: Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.
### Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.

## 6- Performance Evaluation 2: Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.
### All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.