
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct TreeNode
{
	LidarPoint point;
	int id;
	TreeNode* left;
	TreeNode* right;

	TreeNode(LidarPoint setPoint, int setId)
	:	point(setPoint), id(setId), left(NULL), right(NULL)
	{
	}
};

struct KdTree{
    TreeNode* root;

	KdTree():root(NULL)
	{
	}
    void insertHelper(TreeNode** node, uint depth, LidarPoint point, int id)
	{
		if(*node == NULL)
		{
			*node = new TreeNode(point, id);
		}
		else
		{
			uint cd = depth % 3;
			if(cd == 0)
			{
				if(point.x < (*node)->point.x)
				{
					insertHelper(&((*node)->left), depth+1, point, id);
				}
				else
				{
					insertHelper(&((*node)->right), depth+1, point, id);
				}
			}
			if(cd == 1)
			{
				if(point.y < (*node)->point.y)
				{
					insertHelper(&((*node)->left), depth+1, point, id);
				}
				else
				{
					insertHelper(&((*node)->right), depth+1, point, id);
				}
			}
			if(cd == 2)
			{
				if(point.z < (*node)->point.z)
				{
					insertHelper(&((*node)->left), depth+1, point, id);
				}
				else
				{
					insertHelper(&((*node)->right), depth+1, point, id);
				}
			}
		}
	}

	void insert(LidarPoint point, int id)
	{
		insertHelper(&root, 0, point, id);
	}

	void searchHelper(LidarPoint target, TreeNode* node, uint depth, float dist_tol, std::vector<int>& ids)
	{
		if(node != NULL)
		{
			if(node->point.x >= (target.x - dist_tol) && node->point.x <= (target.x + dist_tol)
			&& node->point.y >= (target.y - dist_tol) && node->point.y <= (target.y + dist_tol)
			&& node->point.z >= (target.z - dist_tol) && node->point.z <= (target.z + dist_tol))
			{
				float dx = node->point.x - target.x;
				float dy = node->point.y - target.y;
				float dz = node->point.z - target.z;
				float dist = sqrt(dx*dx + dy*dy + dz*dz);
				if(dist <= dist_tol)
				{
					ids.push_back(node->id);
				}
			}
			uint cd = depth%3;
			if(cd == 0)
			{
				if((target.x - dist_tol) < node->point.x)
				{
					searchHelper(target,node->left, depth+1, dist_tol, ids);
				}
				if((target.x + dist_tol) > node->point.x)
				{
					searchHelper(target, node->right, depth+1, dist_tol, ids);
				}
			}
			if(cd == 1)
			{
				if((target.y - dist_tol) < node->point.y)
				{
					searchHelper(target,node->left, depth+1, dist_tol, ids);
				}
				if((target.y + dist_tol) > node->point.y)
				{
					searchHelper(target, node->right, depth+1, dist_tol, ids);
				}
			}
			if(cd == 2)
			{
				if((target.z - dist_tol) < node->point.z)
				{
					searchHelper(target,node->left, depth+1, dist_tol, ids);
				}
				if((target.z + dist_tol) > node->point.z)
				{
					searchHelper(target, node->right, depth+1, dist_tol, ids);
				}
			}
		}
	}

	std::vector<int> search(LidarPoint target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(target, root, 0, distanceTol, ids);
		return ids;
	}

	void ResourcesFreeHelper(TreeNode** node)
	{
		if(*node)
		{
			return;
		}
		if(!(*node)->right)
		{
			ResourcesFreeHelper(&((*node)->right));
		}
		if(!(*node)->left)
		{
			ResourcesFreeHelper(&((*node)->left));
		}
		delete *node;
	}

	void freeResources()
	{
		ResourcesFreeHelper(&root);
	}
};

#endif /* dataStructures_h */
