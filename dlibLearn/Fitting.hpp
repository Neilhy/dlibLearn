#ifndef Fitting_hpp
#define Fitting_hpp

#include <opencv2/opencv.hpp>
#include "dlib/image_processing/frontal_face_detector.h"
#include <list>
#include <algorithm>
#include <time.h>
#include <string>
#include "mathlib.h"

using namespace std;


static const double smooth_lambda = 10.0;
static const double sparse_lambda = 500.0;
static const double learning_rate = 1e-6;
static const int landmark_num = 68;
static const int iteration_times = 100;
static const int max_rotation_history = 5;
static const int eye_state_change_frame = 3;
static const double eye_change_state_angle_threshold = 0.45;

static int landmark_mapping[68] = {
	0, 2, 4, 6, 8, 10, 12, 14, 16, 15, 13, 11, 9, 7, 5, 3, 1,
	17, 19, 21, 23, 25, 26, 24, 22, 20, 18,
	27, 28, 29, 30,
	31, 33, 35, 34, 32,
	36, 38, 40, 42, 44, 46, 43, 41, 39, 37, 47, 45,
	48, 50, 52, 54, 53, 51, 49, 56, 58, 59, 57, 55, 60, 62, 64, 63, 61, 66, 67, 65
};

static int landmark_weight[68] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1,
	1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

static double arrFace3D[68 * 3] = {
	-61.1943, 19.2675, 17.2794,
	61.1943, 19.2675, 17.2794,
	- 58.6862, 3.11092, 25.5288,
	58.6862, 3.11092, 25.5288,
	- 56.149, - 16.2803, 26.8555,
	56.149, - 16.2803, 26.8555,
	- 54.1547, - 35.0868, 22.457,
	54.1547, - 35.0868, 22.457,
	- 48.524, - 51.2822, 23.2831,
	48.524, - 51.2822, 23.2831,
	- 40.7704, - 64.694, 26.5033,
	40.7704, - 64.694, 26.5033,
	- 29.0407, - 71.7953, 38.1418,
	29.0407, - 71.7953, 38.1418,
	- 17.8277, - 78.3255, 47.1178,
	17.8277, - 78.3255, 47.1178,
	0.0, - 80.8504, 52.297,
	- 52.4505, 35.4669, 34.9537,
	52.4505, 35.4669, 34.9537,
	- 44.8634, 38.4162, 42.3774,
	44.8634, 38.4162, 42.3774,
	- 36.5435, 40.567, 47.1034,
	36.5435, 40.567, 47.1034,
	- 26.2628, 39.7208, 50.5581,
	26.2628, 39.7208, 50.5581,
	- 15.8221, 36.0217, 51.4049,
	15.8221, 36.0217, 51.4049,
	0.0, 24.458, 57.8138,
	0.0, 14.921, 65.1588,
	0.0, 5.10153, 72.3962,
	0.0, - 4.2742, 78.3721,
	- 17.3165, - 17.171, 51.3683,
	17.3165, - 17.171, 51.3683,
	- 9.54034, - 19.9167, 58.0461,
	9.54034, - 19.9167, 58.0461,
	0.0, - 22.1089, 62.1264,
	- 47.9848, 24.5798, 31.2519,
	47.9848, 24.5798, 31.2519,
	- 37.7901, 28.6437, 40.214,
	37.7901, 28.6437, 40.214,
	- 25.7413, 29.6317, 40.4986,
	25.7413, 29.6317, 40.4986,
	- 16.35, 21.4393, 37.1554,
	16.35, 21.4393, 37.1554,
	- 25.5417, 19.4919, 39.4007,
	25.5417, 19.4919, 39.4007,
	- 36.4757, 17.8937, 39.4883,
	36.4757, 17.8937, 39.4883,
	- 26.0263, - 40.0931, 48.1099,
	26.0263, - 40.0931, 48.1099,
	- 18.2356, - 34.7327, 55.8753,
	18.2356, - 34.7327, 55.8753,
	- 9.80985, - 32.4379, 61.2305,
	9.80985, - 32.4379, 61.2305,
	0.0, - 31.9306, 62.7019,
	- 19.7326, - 47.3612, 51.2311,
	19.7326, - 47.3612, 51.2311,
	- 12.4355, - 51.3317, 55.624,
	12.4355, - 51.3317, 55.624,
	0.0, - 54.3237, 55.9109,
	- 16.7967, - 38.6965, 55.1861,
	16.7967, - 38.6965, 55.1861,
	- 8.39335, - 37.8884, 59.6261,
	8.39335, - 37.8884, 59.6261,
	0.0, - 37.2251, 62.4717,
	- 9.94954, - 44.2487, 58.1982,
	9.94954, - 44.2487, 58.1982,
	0.0, - 44.5487, 60.2359,
};

#define FittingInst Fitting::getInst()

class Fitting
{
public:
    static Fitting *getInst();

	void load_blendshapes();
	void load_3DModelPoints();
	
	void calc_camera_matrix(float focal_length, cv::Point2d center);
	void get_2DModelPoints(dlib::full_object_detection &shape);
	size_t get_blendshapesCount() const;
    
	void reset_parameters();
    bool isFirstFitting();
    
    void getRotationAndTranslate(vec3 &arrRotate, vec3 &arrTrans);
    void getFilterRotationAndTranslate(vec3 &arrRotate, vec3 &arrTrans);
    
	void fitting_shape(dlib::full_object_detection &shape);
private:
    Fitting();
    virtual ~Fitting();
    
private:
	std::vector<cv::Point3d> m_blendshapes;
	std::vector<cv::Point3d> m_3DModelPoints;
	std::vector<cv::Point2d> m_2DModelPoints;

	cv::Mat m_cameraMatrix;

	bool m_firstFitting;
    
    vec3 m_vRotate;
    vec3 m_vTranslate;
    
    vec3 m_vFilterRotate;
    vec3 m_vFilterTranslate;
    
    list<vec3> m_vRotateRecord;
    list<vec3> m_vTransRecord;
};

#endif
