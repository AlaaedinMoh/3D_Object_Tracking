
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <iterator>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits> 

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

void FilterOutliers(vector<LidarPoint>& lidarPoints, double tolerance);
vector<vector<LidarPoint>> ClusterLidarPoints(const vector<LidarPoint>& lidarPoints, KdTree* tree, double tolerance);
void ClusteringHelper (int idx, const vector<LidarPoint>& points, vector<LidarPoint>& resultCluster, KdTree* tree, vector<bool>& processed, double distTol);

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow(windowName, topviewImg.rows*0.4, topviewImg.cols*0.4);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
        // cv::imwrite("../Results/Images/Top_View_Img.png", topviewImg);
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double eDist = 0, sum = 0;
    //calculate the mean distance of all matches
    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); itr++)
    {
        sum += itr->distance; 
    }
    eDist = sum/kptMatches.size();
    // std::cout<<"Mean Distance = "<< eDist<<endl;
    //associate the bounding box with keypoints matches
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
    // std::cout<<"Total matches count = "<<kptMatches.size()<<endl;
    // std::cout<<"BB matches count = "<<boundingBox.kptMatches.size()<<endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    cv::Point2f secondPntPrev, secondPntCurr;
    vector<double> relDistances;
    double minDist = 50;
    auto refPntCurr = kptsCurr[kptMatches.begin()->trainIdx].pt;
    auto refPntPrev = kptsPrev[kptMatches.begin()->queryIdx].pt;
    // std::for_each(kptMatches.begin() + 1, kptMatches.end(), [&] (cv::DMatch firstItr)
    // {
    //     firstPntPrev = kptsPrev[firstItr.queryIdx].pt;
    //     firstPntCurr = kptsCurr[firstItr.trainIdx].pt;
    //     // secondPntPrev = kptsPrev[matchItr.queryIdx].pt;
    //     // secondPntCurr = kptsCurr[matchItr.trainIdx].pt;
    //     // std::cout<<"Prev X = "<<secondPntPrev.x<<"\tY = "<<secondPntPrev.y<<endl;
    //     // std::cout<<"Curr X = "<<secondPntCurr.x<<"\tY = "<<secondPntCurr.y<<endl;
    //     // double previousDist = cv::norm(secondPntPrev - firstPntPrev);
    //     // double currentDist = cv::norm(secondPntCurr - firstPntCurr);
    //     // std::cout<<"Prev Dist = "<<previousDist<<"\t, Curr Dist = "<<currentDist<<endl;
    //     // if(previousDist >= minDist && currentDist >= minDist && fabs(previousDist - currentDist) > 0.001)
    //     // {
    //     //     // cout<<"Prev Dist = "<<previousDist<<"\t, Curr Dist = "<<currentDist<<endl;
    //     //     double distRatio = currentDist / previousDist;
    //     //     relDistances.push_back(distRatio);
    //     // }
    //     std::for_each(kptMatches.begin(), kptMatches.end(), [&] (cv::DMatch secondItr)
    //     {
    //         if(firstItr.queryIdx != secondItr.queryIdx)
    //         {
    //             secondPntPrev = kptsPrev[secondItr.queryIdx].pt;
    //             secondPntCurr = kptsCurr[secondItr.trainIdx].pt;
    //             double previousDist = cv::norm(firstPntPrev - secondPntPrev);
    //             double currentDist = cv::norm(firstPntCurr - secondPntCurr);
    //             if(previousDist >= minDist && fabs(previousDist - currentDist) > 0.001)
    //             {
    //                 // cout<<"Prev Dist = "<<previousDist<<"\t, Curr Dist = "<<currentDist<<endl;
    //                 double distRatio = currentDist / previousDist;
    //                 relDistances.push_back(distRatio);
    //             }
    //         }
    //     });
    // });
    std::for_each(kptMatches.begin() + 1, kptMatches.end(), [&] (cv::DMatch matchItr)
    {
        secondPntPrev = kptsPrev[matchItr.queryIdx].pt;
        secondPntCurr = kptsCurr[matchItr.trainIdx].pt;
        // std::cout<<"Prev X = "<<secondPntPrev.x<<"\tY = "<<secondPntPrev.y<<endl;
        // std::cout<<"Curr X = "<<secondPntCurr.x<<"\tY = "<<secondPntCurr.y<<endl;
        double previousDist = cv::norm(secondPntPrev - refPntPrev);
        double currentDist = cv::norm(secondPntCurr - refPntCurr);
        // std::cout<<"Prev Dist = "<<previousDist<<"\t, Curr Dist = "<<currentDist<<endl;
        if(previousDist >= minDist && fabs(previousDist - currentDist) > 0.001)
        {
            // cout<<"Prev Dist = "<<previousDist<<"\t, Curr Dist = "<<currentDist<<endl;
            double distRatio = currentDist / previousDist;
            relDistances.push_back(distRatio);
        }
    });
    if(relDistances.size()==0)
    {
        TTC=0;
        cout<<"TTC = 0"<<endl;
        return;
    }
    std::sort(relDistances.begin(), relDistances.end());
    double med = relDistances.size() % 2 == 0 ? relDistances[relDistances.size()/2] : relDistances[(relDistances.size() + 1)/2];
    // cout<<"Relatives Size = "<<relDistances.size()<<endl;
    // cout<<"Relatives Median = "<<med<<endl;
    double dt = 1/frameRate;
    TTC = -(dt / (1 - med));
    cout<<"Median Camera TTC = "<<TTC<<endl;
    // double sum=0;
    // std::for_each(relDistances.begin(), relDistances.end(),[&] (double dist){
    //     sum+=dist;
    // } );
    // double mean = sum/relDistances.size();
    // // cout<<"Mean = "<<mean<<"\t,Sum = "<<sum<<endl;
    // TTC = -dt/(1 - mean);
    // std::cout<<"Mean Camera TTC = "<<TTC<<endl;
}


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

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // cv::Mat showImg = (currFrame.cameraImg);
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

    // cout<<"Number of boxes in current frame = "<<currFrame.boundingBoxes.size()<<endl;
    // cout<<"Showing the image"<<endl;
    // string windowName = "current frame";
    // cv::resize(showImg, showImg, cv::Size2d(showImg.cols * 2, showImg.rows * 2));
    // // cv::resizeWindow(windowName, showImg.cols*2, showImg.rows*2);
    // cv::imshow("current frame", currFrame.cameraImg);
}