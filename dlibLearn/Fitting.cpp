#include "Fitting.hpp"

Fitting *Fitting::getInst()
{
    static Fitting *s_Fitting = new Fitting();
    return s_Fitting;
}

Fitting::Fitting() : m_firstFitting(true)
{
	load_blendshapes();
	load_3DModelPoints();
}

Fitting::~Fitting()
{
}

size_t Fitting::get_blendshapesCount() const
{
	return m_blendshapes.size() - 1;
}

void Fitting::load_blendshapes()
{
    m_blendshapes.clear();
	for (int i = 0; i < 68; ++i)
	{
		m_blendshapes.push_back(cv::Point3d(arrFace3D[3*i], arrFace3D[3 * i+1], arrFace3D[3 * i+2]));
	}
}

void Fitting::load_3DModelPoints()
{
    m_3DModelPoints.clear();
    
    int iCur = 0;
	for (int i = 0; i < landmark_num; ++i)
	{
		/*if (i >= 48)
		{
			continue;
		}*/
		int index = landmark_mapping[i];
        if(index < 17)
        {
            continue;
        }
        
		m_3DModelPoints.push_back(m_blendshapes[index]);
        
        m_3DModelPoints[iCur].z = -m_3DModelPoints[iCur].z;
        ++iCur;
    }
}

void Fitting::get_2DModelPoints(dlib::full_object_detection &shape)
{
	m_2DModelPoints.clear();

	for (int i = 0; i < landmark_num; ++i)
	{
		/*if (i >= 48)
		{
			continue;
		}*/
        if(i < 17)
        {
            continue;
        }
        
		m_2DModelPoints.push_back(cv::Point2d(shape.part(i).x(), shape.part(i).y()));
	}
}

void Fitting::calc_camera_matrix(float focal_length, cv::Point2d center)
{
	m_cameraMatrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, -focal_length, center.y, 0, 0, 1);
}

bool solvePnPTest(cv::InputArray _opoints, cv::InputArray _ipoints)
{
	cv::Mat opoints = _opoints.getMat(), ipoints = _ipoints.getMat();
	int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
	CV_Assert(npoints >= 0 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)));
	return true;
}

void Fitting::fitting_shape(dlib::full_object_detection &shape)
{
    cv::Mat rotationVector;
    cv::Mat translationVector;
    
	get_2DModelPoints(shape);
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

	cv::solvePnP(cv::Mat(m_3DModelPoints), cv::Mat(m_2DModelPoints), m_cameraMatrix, dist_coeffs, rotationVector, translationVector, false);
    
	m_firstFitting = false;

    double *pRotate = (double *)rotationVector.ptr();
    double *pTrans = (double *)translationVector.ptr();
    
    pRotate[0] *= 180.0f/PI;
    pRotate[1] *= 180.0f/PI;
    pRotate[2] *= 180.0f/PI;
    if(m_vRotateRecord.size() > 0)
    {
        if(abs(pRotate[0]-m_vRotate.x) < 100.0 && abs(pRotate[1]-m_vRotate.y) < 100.0 && abs(pRotate[2]-m_vRotate.z) < 100.0)
        {
            m_vRotate = vec3((float)pRotate[0], (float)pRotate[1], (float)pRotate[2]);
            m_vTranslate = vec3((float)pTrans[0], (float)pTrans[1], (float)pTrans[2]);
        }
    }
    else
    {
        if(abs(pRotate[2]) > 100.0)
        {
            m_vRotate = vec3(0,0,0);
            m_vTranslate = vec3(0,0,-100); //不显示
            m_vFilterRotate = m_vRotate;
            m_vFilterTranslate = m_vTranslate;
            return;
        }
        else
        {
            m_vRotate = vec3((float)pRotate[0], (float)pRotate[1], (float)pRotate[2]);
            m_vTranslate = vec3((float)pTrans[0], (float)pTrans[1], (float)pTrans[2]);
        }
    }
    
    m_vFilterRotate = m_vRotate;
    m_vFilterTranslate = m_vTranslate;
    //对旋转和位移进行过滤
    while(m_vRotateRecord.size() >= 10)
    {
        m_vRotateRecord.pop_front();
    }
    m_vRotateRecord.push_back(m_vFilterRotate);
    
    while(m_vTransRecord.size() >= 10)
    {
        m_vTransRecord.pop_front();
    }
    m_vTransRecord.push_back(m_vFilterTranslate);
    
    
    size_t iStart = 0;
    if(m_vRotateRecord.size() > 3)
    {
        iStart =m_vRotateRecord.size()-3;
    }
    vec3 vRotateMax = vec3(-100000.0f, -100000.0f, -100000.0f);
    vec3 vRotateMin = vec3(100000.0f, 100000.0f, 1000000.0f);
    vec3 vTransMax = vec3(-100000.0f, -100000.0f, -100000.0f);
    vec3 vTransMin = vec3(100000.0f, 100000.0f, 1000000.0f);
    vec3 vRotate = vec3(0,0,0);
    vec3 vTrans = vec3(0,0,0);
    list<vec3>::iterator it = m_vRotateRecord.begin();
    while(it != m_vRotateRecord.end())
    {
        vec3 vRT = (*it);
        if(iStart == 0)
        {
            vRotate += vRT;
        }
        else
        {
            --iStart;
        }
        
        if(vRT.x > vRotateMax.x){vRotateMax.x = vRT.x;}
        if(vRT.y > vRotateMax.y){vRotateMax.y = vRT.y;}
        if(vRT.z > vRotateMax.z){vRotateMax.z = vRT.z;}
        
        if(vRT.x < vRotateMin.x){vRotateMin.x = vRT.x;}
        if(vRT.y < vRotateMin.y){vRotateMin.y = vRT.y;}
        if(vRT.z < vRotateMin.z){vRotateMin.z = vRT.z;}
        ++it;
    }
    vRotate /= fmin(m_vRotateRecord.size(),3);
    if(m_vRotateRecord.size() >= 10)
    {
        //求取最大移动距离
        //vec3 vDRotate = (vRotateMax-vRotateMin)/m_vRotateRecord.size();
    }
    m_vFilterRotate = vRotate;
    
    iStart = 0;
    if(m_vRotateRecord.size() > 3)
    {
        iStart =m_vRotateRecord.size()-3;
    }
    it = m_vTransRecord.begin();
    while(it != m_vTransRecord.end())
    {
        vec3 vTT = (*it);
        if(iStart == 0)
        {
            vTrans += vTT;
        }
        else
        {
            --iStart;
        }
        
        if(vTT.x > vTransMax.x){vTransMax.x = vTT.x;}
        if(vTT.y > vTransMax.y){vTransMax.y = vTT.y;}
        if(vTT.z > vTransMax.z){vTransMax.z = vTT.z;}
        
        if(vTT.x < vTransMin.x){vTransMin.x = vTT.x;}
        if(vTT.y < vTransMin.y){vTransMin.y = vTT.y;}
        if(vTT.z < vTransMin.z){vTransMin.z = vTT.z;}
        ++it;
    }
    vTrans /= fmin(m_vTransRecord.size(),3);
    if(m_vTransRecord.size() >= 10)
    {
        vec3 vDTrans = (vTransMax-vTransMin)/m_vTransRecord.size();
        vDTrans.x *= vDTrans.x;
        vDTrans.y *= vDTrans.y;
        vDTrans.z *= vDTrans.z;
        
        vDTrans /= 10.0f;
        vDTrans.x = fmin(vDTrans.x, 1.0f);
        vDTrans.y = fmin(vDTrans.y, 1.0f);
        vDTrans.z = fmin(vDTrans.z, 1.0f);
        
        m_vFilterTranslate.x = vTrans.x*(1.0f-vDTrans.x) + m_vFilterTranslate.x*vDTrans.x;
        m_vFilterTranslate.y = vTrans.y*(1.0f-vDTrans.y) + m_vFilterTranslate.y*vDTrans.y;
        m_vFilterTranslate.z = vTrans.z*(1.0f-vDTrans.z) + m_vFilterTranslate.z*vDTrans.z;
    }
    else
    {
        m_vFilterTranslate = vTrans;
    }
}

void Fitting::getRotationAndTranslate(vec3 &arrRotate, vec3 &arrTrans)
{
    arrRotate = m_vRotate;
    arrTrans = m_vTranslate;
}

void Fitting::getFilterRotationAndTranslate(vec3 &arrRotate, vec3 &arrTrans)
{
    arrRotate = m_vFilterRotate;
    arrTrans = m_vFilterTranslate;
}

bool Fitting::isFirstFitting()
{
    return m_firstFitting;
}

void Fitting::reset_parameters()
{
    m_vTransRecord.clear();
    m_vRotateRecord.clear();
    
    m_vFilterRotate = m_vRotate;
    m_vFilterTranslate = m_vTranslate;
    
	m_firstFitting = true;
}
