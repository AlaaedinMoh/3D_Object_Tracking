
#include <numeric>
#include "matching2D.hpp"

using namespace std;
using namespace cv;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<DMatch>> knnMatches;
        matcher->knnMatch(descSource,descRef, knnMatches, 2);
        float thresh_ratio = 0.8;
        for(size_t i = 0; i < knnMatches.size(); i++)
        {
            if(knnMatches[i][0].distance < thresh_ratio * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
         Ptr<cv::AKAZE> aExtractor = cv::AKAZE::create();
        double tt = (double)cv::getTickCount();
    aExtractor->compute(img, keypoints, descriptors);
    tt = ((double)cv::getTickCount() - tt) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * tt / 1.0 << " ms" << endl;
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        return;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int neighborhoodSize = 4;
    int sobelKernel = 3;
    double freeParam = 0.05;
    double t = (double)cv::getTickCount();
    Mat HaarisRslt = Mat::zeros(img.size(), CV_32FC1);
    cornerHarris(img,HaarisRslt,neighborhoodSize, sobelKernel, freeParam);
    Mat dst_norm,dst_norm_scaled;
    normalize( HaarisRslt, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    uint thresh = 70;
    for(int row = 0; row < dst_norm.rows; row++)
    {
        for(int col = 0; col < dst_norm.cols; col++)
        {
            if((int)dst_norm.at<float>(row, col) > thresh)
            {
                circle( dst_norm_scaled, Point( col, row ), 10,  Scalar(0), 2, 8, 0 );
                KeyPoint newKp;
                newKp.pt = Point2f(col, row);
                newKp.size=neighborhoodSize;
                keypoints.push_back(newKp);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HAARIS Corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    Ptr<cv::DescriptorExtractor> detector;
    if(detectorType.compare("FAST") == 0)
    {
        detector = FastFeatureDetector::create();
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = ORB::create();
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    if(detector)
    {
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType <<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else
    {
        cout<<"Could not define the detector type!!"<<endl;
    }
}