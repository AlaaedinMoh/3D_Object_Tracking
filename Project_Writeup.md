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
            sumPrev+=prevPnt.x;
            prevCount++;
        }
    });
    std::for_each(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](LidarPoint currPnt)
    {
        if(abs(currPnt.y)<=halfLaneWidth)
        {
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
The code shrink the ROI to avoid the addind key points matching from the road. Then it filters out the keypoints depending on the matche distance and the spatial distance between the previous and the current frames.
```C++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
  //associate the bounding box with keypoints matches
  double widthScaleFactor = 0.8;
  double heightScaleFactor = 0.8;
  double scaledWidth = boundingBox.roi.width * widthScaleFactor;
  double scaledHeight = boundingBox.roi.height * heightScaleFactor;
  cv::Rect scaledRoi = cv::Rect(boundingBox.roi.x + (boundingBox.roi.width - scaledWidth) / 2, boundingBox.roi.y + (boundingBox.roi.height - scaledHeight) / 2, scaledWidth, scaledHeight);
  boundingBox.roi = scaledRoi;

  for (auto & kptMatche : kptMatches) {
    cv::KeyPoint prevKp = kptsPrev[kptMatche.queryIdx];
    cv::KeyPoint currKp = kptsCurr[kptMatche.trainIdx];

    if (scaledRoi.contains(prevKp.pt) && scaledRoi.contains(currKp.pt) )
    {
      boundingBox.kptMatches.push_back(kptMatche);
    }
  }
}
```

## 4- Compute Camera-based TTC: Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.
Calculating the camera TTC is done by measuring the relative distances between the keypoints in the ROI.
```C++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
  vector<double> relDistances{};
  double minDist = 100;

    if(kptMatches.size() < 2)
    {
        cout<<"No matching points. Camera TTC = 0"<<endl;
        TTC = NAN;
        return;
    }

  std::for_each(kptMatches.begin() + 1, kptMatches.end(), [&](cv::DMatch firstItr) {
    auto refPntPrev = kptsPrev[firstItr.queryIdx].pt;
    auto refPntCurr = kptsCurr[firstItr.trainIdx].pt;

    std::for_each(kptMatches.begin() + 1, kptMatches.end(), [&](cv::DMatch matchItr) {
      auto secondPntPrev = kptsPrev[matchItr.queryIdx].pt;
      auto secondPntCurr = kptsCurr[matchItr.trainIdx].pt;

      double previousDist = cv::norm(secondPntPrev - refPntPrev);
      double currentDist = cv::norm(secondPntCurr - refPntCurr);

      if (previousDist > minDist && std::abs(previousDist - currentDist) > 0.001) {
        double distRatio = currentDist / previousDist;
        relDistances.push_back(distRatio);
      }
    });
  });
  if (relDistances.empty()) {
    TTC = NAN;
    return;
  }
  std::sort(relDistances.begin(), relDistances.end());
  double med = relDistances.size() % 2 == 0 ? relDistances[relDistances.size() / 2] : relDistances[(relDistances.size() + 1) / 2];
  double dt = 1 / frameRate;
  TTC = -dt / (1 - med);
  cout << "Median Camera TTC = " << TTC << endl;
}
```
## Preformance Evaluation:
To evaluate the performance I created a class that can save the results in CSV-file.

## Evaluating Performance:
To evaluate the performance I created a new class that write CSV files and save the performance data in the file. I choosed to save the number of the total detected key points and the elapsed detecting time in one file. The TTC of lidar and the camera and the difference between them are saved in another file.
```C++
class CSV_Writer
{
private:
    //Column name - column value type
    map<int, string, std::less<int>> _csvCols;
    vector<CSV_Line> _contents;
    string _separator;
    void AddContentsToStream(map<int, string> lineContents, fstream& currentStream);
public:
    CSV_Writer(map<int, string> colsNames, string separator);
    ~CSV_Writer();
    void AddLine(const CSV_Line& line);
    bool SaveFile(string filePath);
};

CSV_Writer::CSV_Writer(map<int, string> cols, string separator = ",")
{
    _separator = separator;
    for(const auto& col : cols)
    {
        _csvCols.insert(col);
    }

}

CSV_Writer::~CSV_Writer()
{
}

void CSV_Writer::AddLine(const CSV_Line& line)
{
    _contents.push_back(line);
}

bool CSV_Writer::SaveFile(string filePath)
{
    try
    {
        bool existed = false;
        ifstream file(filePath);
        if(file)
        {
            existed = true;
            file.close();
        }
        fstream fileStream (filePath, std::ios_base::app | std::ios_base::out);
        string lineTxt = "";
        if(existed)
        {
            for(const CSV_Line& line : _contents)
            {
                AddContentsToStream(line.lineMap, fileStream);
            }
        }
        else
        {
            for(const auto& col : _csvCols)
            {
                lineTxt += col.second + _separator;
            }
            lineTxt+="\n";
            fileStream<<lineTxt;
            for(const CSV_Line& line : _contents)
            {
                AddContentsToStream(line.lineMap, fileStream);
            }
        }
        lineTxt = "";
        for(const auto& col : _csvCols)
        {
            lineTxt += " " + _separator;
        }
        lineTxt+="\n";
        fileStream<<lineTxt;
        fileStream.close();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void CSV_Writer::AddContentsToStream(map<int, string> lineContents, fstream& currentStream)
{
    map<int, string, std::less<int>> orderedMap;
    for(const auto& line : lineContents)
    {
        orderedMap.insert(make_pair(line.first, line.second));
    }
    string lineTxt = "";
    for(const auto& entry : orderedMap)
    {
        lineTxt += entry.second + _separator;
    }
    lineTxt += "\n";
    currentStream<<lineTxt;
}
```

## 5- Performance Evaluation 1: Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.
### Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.
|Scenario|Detector-Descriptor|Frame|TTC-Lidar|TTC-Camera|Differnce|
|:-------|:-----------------:|:---:|:-------:|:--------:|:-------:|
|1|ORB-BRISK|Frame1|12.2891|nan|nan|
|2|SHITOMASI-BRIEF|Frame3|16.384452|8.729978|7.654474|
After the observation of the results, we can come to the conclusion, that the TTC-Camera is not very reliable. Because it depends on the detector-descriptor types. Each one of them can give us different results, which could lead to inaccurate estimation. More than that, camera TTC depends on the estimating the distance between the pixels, but the lidar gives us the measuments of the distance directly. However, detecting keypoints depends on the surrounding environment. This means, that for example in the darkness, it is impossible to estimate the camera TTC because there is no light to detect the keypoints. Or in foggy weather the camera could not detect the other cars very well. The technical specifications of the camera itself could play a role detecting the keypoints also.


## 6- Performance Evaluation 2: Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.
### All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs.
|Scenario|Detector-Descriptor|Frame|TTC-Lidar|TTC-Camera|Differnce|
|:-------|:-----------------:|:---:|:-------:|:--------:|:-------:|
|1|SHITOMASI-BRISK|Frame16|8.898673|8.574856|0.323817|
|2|FAST-BRIEF|Frame15|8.3212|8.443254|0.122054|
|3|BRISK-BRISK|	Frame6|	13.751074|	13.609277|	0.141797|
In some scenarien there were very close estimations between the Lidar and the camera. The table above illustrate some samples of these results. Anyway the combination of BRISK detector and ORB descriptor has the largest number of close etimations(You can find it in the file estimation CSV file [here](./Results/TTC_Estimate_Diff.csv)).