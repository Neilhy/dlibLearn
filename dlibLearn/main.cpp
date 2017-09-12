#include<dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include<dlib/gui_widgets.h>
#include<dlib/image_io.h>

#include<dlib/svm_threaded.h>
#include<dlib/image_processing.h>
#include<dlib/data_io.h>
#include<dlib/opencv.h>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/tracking/tracker.hpp>
#include<opencv2/tracking.hpp>


#include<iostream>
#include<fstream>
#include<string>
#include <sstream>

#include"Fitting.hpp"
#include"tinyxml2.h"
using namespace dlib;
using namespace std;


template<class T>
int getArrayLen(T& array) {
	return (sizeof(array) / sizeof(array[0]));
}

void detectFaces() {
	try
	{

		frontal_face_detector detector = get_frontal_face_detector();
		image_window win;
		int imgNum = 4;
		string filename[] = { "E:/Chrome/face1.jpg","E:/Chrome/faces.jpg","E:/001Company/face/dlib-19.2/examples/faces/2008_004176.jpg","E:/001Company/face/dlib-19.2/examples/faces/dogs.jpg" };
		// Loop over all the images provided on the command line.
		for (int i = 0; i < imgNum; ++i)
		{
			cout << "processing image " << filename[i] << endl;
			array2d<unsigned char> img;
			//array2d<rgb_pixel>img;
			load_image(img, filename[i]);

			//cout << "img rows(height): " << img.nr() << " cols(weight): " << img.nc() << endl;
			pyramid_down<2> pd;
			pyramid_up(img,pd);
			cout <<"nc: "<< img.nc() << " nr: " << img.nr() << endl;

			std::vector<rectangle> dets = detector(img);

			cout << "Number of faces detected: " << dets.size() << endl;

			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(dets, rgb_pixel(255, 0, 0));

			cout << "Hit enter to process the next image..." << endl;
			cin.get();
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

void trainingDetector() {
	try
	{

	const string faces_train_directory = "E:/001Company/face/dlib-19.2/examples/faces";
	//const string faces_test_directory = "E:/dlibTrain";
	
	dlib::array<array2d<unsigned char>> images_train;//, images_test
	std::vector<std::vector<rectangle>> face_boxes_train;//, face_boxes_test;
	cout << "Before loading images..." << endl;
	//load_image_dataset(images_train, face_boxes_train, faces_train_directory + "/fontFaces_no_part_samples_test.xml");
	load_image_dataset(images_train, face_boxes_train, faces_train_directory + "/training.xml");
	//load_image_dataset(images_test, face_boxes_test, faces_test_directory + "/testing.xml");
	cout << "After loading images..." << endl;

	//upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);
	//upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
	//dlib::rotate_image_dataset()// pi * 27/180
	
	//add_image_left_right_flips(images_train, face_boxes_train);
	cout << "num training images: " << images_train.size() << endl;
	//cout << "num testing images:  " << images_test.size() << endl;

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	image_scanner_type scanner;
	scanner.set_nuclear_norm_regularization_strength(1.0);//too big will not fit data data//nuclear norm regularizer: 9
	//configure_nuclear_norm_regularizer();//nuclear norm regularizer: 9 ??
	scanner.set_detection_window_size(60,60);//minimize window 50,50 ？
	scanner.set_cell_size(8);
	scanner.set_padding(0);
	
	structural_object_detection_trainer<image_scanner_type> trainer(scanner);

	trainer.set_num_threads(4);//
	trainer.set_c(1);// too small will not fit the data//700
	// We can tell the trainer to print it's progress to the console if we want.  
	trainer.be_verbose();
	trainer.set_epsilon(0.01);//precision//0.05
	trainer.set_loss_per_missed_target(1);
	
	//std::vector<std::vector<rectangle>> removeRects = remove_unobtainable_rectangles(trainer, images_train, face_boxes_train);
	
	cout << "Before training images..." << endl;
	object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);
	detector = threshold_filter_singular_values(detector, 0.15);//singular value threshold: 0.15
	serialize("test.svm") << detector;
	dlib::matrix<double, 1, 3> m= test_object_detection_function(detector, images_train, face_boxes_train);
	cout << "training results: " <<  m << endl;//testing  precision, recall, average precision
	//ofstream ofile; ofile.flush();ofile.close()
	//ofile.open("test.txt");
	//ofile << "hahah\n" << 123;
	//ofile << "nihao\n" << endl;
	//ofile << m;
	//ofile.close();
	//cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;//testing

	//image_window hogwin(draw_fhog(detector), "First detector of ......");

	//image_window win;
	//for (unsigned long i = 0; i < images_test.size(); ++i)
	//{
		// Run the detector and get the face detections.
		//std::vector<rectangle> dets = detector(images_test[i]);

		//std::vector<rectangle> remo_Rec = removeRects[i];

	//	win.clear_overlay();
	//	win.set_image(images_test[i]);
	//	win.add_overlay(dets, rgb_pixel(255, 0, 0));
	//	//win.add_overlay(remo_Rec, rgb_pixel(255, 255, 0));

	//	cout << "Hit enter to process the next image..." << endl;
	//	cin.get();
	//}



	// Then you can recall it using the deserialize() function.
	//object_detector<image_scanner_type> detector2;
	//deserialize("face_detector_first.svm") >> detector2;
		
	//// You can see how many separable filters are inside your detector like so:
	//cout << "num filters: " << num_separable_filters(detector) << endl;//num filters: 78
	//// You can also control how many filters there are by explicitly thresholding the
	//// singular values of the filters like this:
	//detector = threshold_filter_singular_values(detector, 0.1);//singular value threshold: 0.15

	/*std::vector<object_detector<image_scanner_type> > my_detectors;
	my_detectors.push_back(detector);
	std::vector<rectangle> dets = evaluate_detectors(my_detectors, images_train[0]);
*/
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}
void t() {
	try
	{

		const string faces_train_directory = "E:/dlibTrain";

		dlib::array<array2d<unsigned char>> images_train;
		std::vector<std::vector<rectangle>> face_boxes_train;
		cout << "Before loading images..." << endl;
		load_image_dataset(images_train, face_boxes_train, faces_train_directory + "/fontFaces_no_part_samples_test.xml");
		cout << "After loading images..." << endl;

		upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);

		const double angle = 3.1415926 * 27 / 180;
		dlib::rotate_image_dataset(angle, images_train, face_boxes_train);// pi * 27/180 左旋
		cout << "num training images: " << images_train.size() << endl;

		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
		image_scanner_type scanner;
		scanner.set_nuclear_norm_regularization_strength(1.0);		
		scanner.set_detection_window_size(80, 80);//minimize window
		scanner.set_cell_size(8);
		scanner.set_padding(0);

		structural_object_detection_trainer<image_scanner_type> trainer(scanner);

		trainer.set_num_threads(4);
		trainer.set_c(700);
		trainer.be_verbose();
		trainer.set_epsilon(0.05);//precision
		trainer.set_loss_per_missed_target(1);
		string f[] = { "haha","haha" };
		string t = f[0] + "a ";
		cout << "Before training images..." << endl;
		object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);
		detector = threshold_filter_singular_values(detector, 0.15);//singular value threshold: 0.15
		serialize("fontface_detector_no_part_samples_test.svm") << detector;

		cout << "training results: " << test_object_detection_function(detector, images_train, face_boxes_train) << endl;//testing  precision, recall, average precision
																														
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}
void tt() {
	try
	{
		
		const string faces_train_xml = "E:/001Company/face/dlib-19.2/examples/faces/training.xml";
		const string faces_test_xml = "E:/001Company/face/dlib-19.2/examples/faces/testing.xml";

		int C[] = { 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200 };
		float nuclear[] = { 1.0, 1.2, 1.4, 0.8, 0.6, 0.4, 0.2, 0.1 };
		float thre[] = { 0.12, 0.10, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25 };
		int degree[] = { 27, -27 };
		string fileName[] = { "leftRotateface_detector_C","rightRotateface_detector_C" };
		int C_len = getArrayLen(C), nuclear_len = getArrayLen(nuclear), thre_len = getArrayLen(thre), degree_len = getArrayLen(degree);
		ofstream of;

		

		for (int d = 0; d<degree_len; d++) {

			int degree_ = degree[d];
			double angle = 3.1415926 * degree_ / 180;

			dlib::array<array2d<unsigned char>> images_train;
			std::vector<std::vector<rectangle>> face_boxes_train, face_boxes_test;
			cout << "Before loading images..." << endl;
			load_image_dataset(images_train, face_boxes_train, faces_train_xml);
			cout << "num training images: " << images_train.size() << endl;
			dlib::rotate_image_dataset(angle, images_train, face_boxes_train);
			cout << "Rotated :" << degree[d] << endl;

			dlib::array<array2d<unsigned char>> images_test;
			load_image_dataset(images_test, face_boxes_test, faces_test_xml);
			cout << "num testing images: " << images_test.size() << endl;
			dlib::rotate_image_dataset(angle, images_test, face_boxes_test);

			for (int i = 0; i<C_len; i++) {
				for (int j = 0; j<thre_len; j++) {
					for (int k = 0; k<nuclear_len; k++) {

						typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size; 7,8,9 no
						image_scanner_type scanner;
						scanner.set_nuclear_norm_regularization_strength(nuclear[k]);
						scanner.set_detection_window_size(60,60);//minimize window ?????要统一为80才可以顺利拼接。。。
						scanner.set_cell_size(8);
						scanner.set_padding(0);

						structural_object_detection_trainer<image_scanner_type> trainer(scanner);

						trainer.set_num_threads(12);
						trainer.set_c(C[i]);
						//trainer.be_verbose();
						trainer.set_epsilon(0.05);//precision
						trainer.set_loss_per_missed_target(1);

						cout << "Before training images..." << endl;
						object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);
						detector = threshold_filter_singular_values(detector, thre[j]);//singular value threshold
						cout << "After training images..." << endl;
						stringstream ss;
						ss.clear();
						ss << fileName[d] << C[i] << "_nuclear" << nuclear[k] << "_thre" << thre[j] << "_A_H_L_I_Merge.svm";
						serialize(ss.str()) << detector;
						of.open("trainList", ios::app);
						of << ss.str() << "\n";
						dlib::matrix<double, 1, 3> mm = test_object_detection_function(detector, images_train, face_boxes_train);
						of << "\ntraining results: " << mm << "\n";
						of.flush();
						cout << "training results: " << mm << endl;//testing  precision, recall, average precision

						
						mm = test_object_detection_function(detector, images_test, face_boxes_test);
						of << "testing results:  " << mm << "\n\n";
						of.close();
						cout << "testing results: " << mm << endl;

					}
				}
			}
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}
void test_face_detector_L_Detector() {
	frontal_face_detector detectorStandard = get_frontal_face_detector();

	const string faces_test_directory = "E:/dlibTrain";

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	object_detector<image_scanner_type> detector;
	deserialize((faces_test_directory+"/fontface_detector_no_part_samples_test.svm").c_str()) >> detector;

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<rectangle>> face_boxes_test;
	load_image_dataset(images_test, face_boxes_test, faces_test_directory + "/300trainAFW.xml");

	//upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
	cout << "num testing images:  " << images_test.size() << endl;
	//cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;//testing
	//dlib::rotate_image_dataset(3.1415926*27 / 180, images_test, face_boxes_test);// pi * 27/180  左旋27度

	image_window winStandard,win;
	for (unsigned long i = 0; i < images_test.size(); ++i)
	{
		std::vector<rectangle> detsStandard = detectorStandard(images_test[i]);
		std::vector<rectangle> dets = detector(images_test[i]);
		//std::vector<rectangle> dets = face_boxes_test[i];

		winStandard.clear_overlay();
		winStandard.set_image(images_test[i]);
		winStandard.add_overlay(detsStandard, rgb_pixel(0, 255, 0));

		win.clear_overlay();
		win.set_image(images_test[i]);
		win.add_overlay(dets, rgb_pixel(255, 0, 0));
		cout << "Hit enter to process the next image..." << endl;
		cin.get();
	}
}
void test_face_detector_A_L_I_Detector() {
	frontal_face_detector detectorStandard = get_frontal_face_detector();

	const string faces_test_directory = "E:/dlibTrain";

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	object_detector<image_scanner_type> detector;
	deserialize("face_detector_A_L_I.svm") >> detector;

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<rectangle>> face_boxes_test;

	load_image_dataset(images_test, face_boxes_test, faces_test_directory+ "/300someHELEN.xml");
	//dlib::rotate_image_dataset(3.1415926, images_test, face_boxes_test);//angle = 弧度制,3.1415926=180度

	//upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
	cout << "num testing images:  " << images_test.size() << endl;
	//cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;//testing

	image_window winStandard, win;
	for (unsigned long i = 0; i < images_test.size(); ++i)
	{
		std::vector<rectangle> detsStandard = detectorStandard(images_test[i]);
		std::vector<rectangle> dets = detector(images_test[i]);
		//std::vector<rectangle> dets = face_boxes_test[i];

		winStandard.clear_overlay();
		winStandard.set_image(images_test[i]);
		winStandard.add_overlay(detsStandard, rgb_pixel(0, 255, 0));

		win.clear_overlay();
		win.set_image(images_test[i]);
		win.add_overlay(dets, rgb_pixel(255, 0, 0));
		cout << "Hit enter to process the next image..." << endl;
		cin.get();
	}
}
void test_face_detector_L_Detector_Camera() {
	cv::VideoCapture cap(0);
	//cv::VideoCapture cap("E:/software/Youku Files/download/test.mp4");
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return;
	}
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size

	std::vector<object_detector<image_scanner_type>> my_detectors;
	object_detector<image_scanner_type> detector_front,detector_left,detector_right, detector_left_rotate, detector_right_rotate;
	deserialize("fontface_detector_C1000_nuclear0.2_A_H_L_I_Merge.svm") >> detector_front;
	deserialize("leftface_detector_C500_nuclear0.8_A_H_L_I_Merge.svm") >> detector_left;
	deserialize("rightface_detector_C600_nuclear1.0_thre0.22_A_H_L_I_Merge.svm") >> detector_right;
	deserialize("leftRotateface_detector_C650_nuclear0.2_thre0.16_8080_A_H_L_I_Merge.svm") >> detector_left_rotate;
	deserialize("rightRotateface_detector_C900_nuclear0.8_thre0.1_8080_A_H_L_I_Merge.svm") >> detector_right_rotate;
	
	my_detectors.push_back(detector_front);//集体测试
	my_detectors.push_back(detector_left);
	my_detectors.push_back(detector_right);
	my_detectors.push_back(detector_left_rotate);
	my_detectors.push_back(detector_right_rotate);

	//object_detector<image_scanner_type> detector_all(my_detectors);//将检测器数组整合为一个总检测器
	//serialize("detector_all.svm") << detector_all;

	object_detector<image_scanner_type> detector_all;
	deserialize("detector_all.svm") >> detector_all;

	//object_detector<image_scanner_type> detector;
	//deserialize("E:/dlibTrain/leftRotateface_detector_C700_nuclear0.2_thre0.08_A_H_L_I_Merge.svm") >> detector;//单个测试

	frontal_face_detector standardDetector = get_frontal_face_detector();
	//my_detectors.push_back(standardDetector);//把标准的也加进来

	//image_window win;
	//win.set_title("A_H_L_I_Merge");
	image_window standardWin;
	standardWin.set_title("standard front");
	image_window allWin;
	allWin.set_title("all detector");
	while (cv::waitKey(1) != 27)
	{
		// Grab a frame  
		cv::Mat temp;
		cap >> temp;
		cv::flip(temp, temp, 1);
		//double scale = 2;
		//cv::Size dsize = cv::Size(temp.cols*scale, temp.rows*scale);
		//cv::Mat tmp = cv::Mat(dsize, temp.type());
		//cv::resize(temp, tmp, dsize);
		cv_image<bgr_pixel> dlib_img(temp);
		
		//std::vector<rectangle> dets = detector(dlib_img);//单个检测
		//std::vector<rectangle> detsVec = evaluate_detectors(my_detectors, dlib_img);//集体检测
		std::vector<rectangle> detsAll = detector_all(dlib_img);//整合体检测
		std::vector<rectangle> sdets = standardDetector(dlib_img);//标准检测

		//win.clear_overlay();
		//win.set_image(dlib_img);
		//win.add_overlay(detsVec, rgb_pixel(255, 0, 0));

		allWin.clear_overlay();
		allWin.set_image(dlib_img);
		allWin.add_overlay(detsAll, rgb_pixel(0, 0, 255));
		
		standardWin.clear_overlay();
		standardWin.set_image(dlib_img);
		standardWin.add_overlay(sdets, rgb_pixel(0, 255, 0));
	}
}
int test_face_detector() {
	
	const string faces_svm = "E:/dlibTrain/fontface_detector_A_H_L_I_Merge.svm"; 
	const string faces_test_xml = "E:/dlibTrain/fontFaces_H_testset_no_missed_faces_no_part.xml"; 

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	object_detector<image_scanner_type> detector;
	deserialize(faces_svm) >> detector;

	frontal_face_detector detectorStandard = get_frontal_face_detector();

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<rectangle>> face_boxes_test;
	load_image_dataset(images_test, face_boxes_test, faces_test_xml);

	upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
	cout << "num testing images:  " << images_test.size() << endl;
	cout << "standard testing results:  " << test_object_detection_function(detectorStandard, images_test, face_boxes_test) << endl;//testing
	cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;//testing
	return 0;
}
void darkenHalfImage() {
	const string faces_test_directory = "E:/dlibTrain";
	dlib::array<array2d<unsigned char>> imagesArray;
	std::vector<std::vector<rectangle>> boxesArray;

	load_image_dataset(imagesArray, boxesArray, faces_test_directory + "/fontFaces_samples_test.xml");
	cv::Mat src;
	std::vector<rectangle> dets;
	rectangle box;
	for (unsigned long i = 0; i < imagesArray.size(); i++) {
		src = dlib::toMat(imagesArray[i]);
		dets = boxesArray[i];
		for (int j = 0; j < dets.size(); j++) {
			box = dets[j];
			cv::Mat dark(box.height(), box.width()*0.4, CV_8UC1,cv::Scalar(0));
			dark.copyTo(src(cv::Rect(box.left(), box.top(), dark.cols, dark.rows)));
			cv::imshow("dest", src);
			cin.get();
			//string file = (faces_test_directory +"/"+ to_string(i)+".jpg");
			//char*filename = new char[file.length() + 1];
			//strcpy(filename, file.c_str());
			//cv::imwrite(filename, src);
		}
	}
}

void faceRotate() {
	const string faces_test_directory = "E:/dlibTrain";

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<full_object_detection>> face_boxes_test;
	load_image_dataset(images_test, face_boxes_test, faces_test_directory + "/test.xml");
	FittingInst->load_blendshapes();
	FittingInst->load_3DModelPoints();
	for (int i = 0; i < images_test.size(); i++) {
		double width = images_test[i].nc();//cols = width
		double height = images_test[i].nr();
		FittingInst->calc_camera_matrix((width < height ? width : height), cv::Point2d(width / 2, height / 2));
		//cv::imshow("face", toMat(images_test[i]));
		//cv::waitKey(0);
		std::vector<full_object_detection> fods = face_boxes_test[i];
		for (int j = 0; j < fods.size(); j++) {
			FittingInst->fitting_shape(fods[j]);
			vec3 vRotate;
			vec3 vTrans;
			FittingInst->getRotationAndTranslate(vRotate, vTrans);
			cout << "Rotate x: " << vRotate.x << " y: " << vRotate.y << " z: " << vRotate.z << endl;
			cout<<"Trans x: "<<vTrans.x<< " y: " << vTrans.y << " z: " << vTrans.z << endl<<endl;
			//cin.get();
		}
	}
}

void cameraFaceRotate() {
	try {
		cv::VideoCapture cap(0);
		//cv::VideoCapture cap("E:/software/Youku Files/download/test.mp4");
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return ;
		}

		// Load face detection and pose estimation models.  
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		FittingInst->load_blendshapes();
		FittingInst->load_3DModelPoints();

		//image_window win;

		// Grab and process frames until the main window is closed by the user.  
		//int num = 1000;
		//cap.set(cv::CAP_PROP_POS_FRAMES, 240);//跳过开头的240帧
		while (cv::waitKey(1) != 27)
		{
			// Grab a frame  
			cv::Mat temp;
			cap >> temp;
			cv::flip(temp, temp, 1); 
			
			double width = temp.cols;//cols = width
			double height = temp.rows;
			FittingInst->calc_camera_matrix((width < height ? width : height), cv::Point2d(width / 2, height / 2));

			cv_image<bgr_pixel> dlib_img(temp);

			// Detect faces   
			std::vector<rectangle> faces = detector(dlib_img);
			// Find the pose of each face.  
			std::vector<full_object_detection> shapes;
			ostringstream oss;
			string infoText;
			for (unsigned long i = 0; i < faces.size(); ++i) {
				vec3 vRotate;
				vec3 vTrans;
				full_object_detection fod= pose_model(dlib_img, faces[i]);
				shapes.push_back(fod);
				FittingInst->fitting_shape(fod);
				FittingInst->getRotationAndTranslate(vRotate, vTrans);
				oss << " x : " << vRotate.x ;
				infoText = oss.str();
				cv::putText(temp, infoText, cv::Point(25 * (i + 1), 25), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0));
				oss.str("");
				
				oss << " y : " << vRotate.y ;
				infoText = oss.str();
				cv::putText(temp, infoText, cv::Point(25 * (i + 1), 75), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0));
				oss.str("");

				oss << " z : " << vRotate.z;
				infoText = oss.str();
				cv::putText(temp, infoText, cv::Point(25 * (i + 1), 125), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255));
				oss.str("");
			}
			if (!shapes.empty()) {
				for (int i = 0; i < 68; i++) {
					cv::circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);//  shapes[0].part(i).x();//68个  
					
				}
			}

			//win.clear_overlay();
			//win.set_image(dlib_img);
			//win.add_overlay(render_face_detections(shapes));
			//Display it all on the screen  

			imshow("Face rotation", temp);
			//num--;
			//if (num == 0) {
				//cin >> num;
			//}
		}
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

void divide_Rotations() {//按照角度分类
	const string faces_test_directory = "E:/dlibTrain";
	tinyxml2::XMLDocument doc, fontFace;
	tinyxml2::XMLElement* root = fontFace.NewElement("dataset");
	fontFace.InsertFirstChild(root);
	root = fontFace.NewElement("images");
	fontFace.RootElement()->InsertFirstChild(root);

	doc.LoadFile((faces_test_directory + "/300all_has_Part_H_testset_All.xml").c_str());
	cout << "loaded xml.." << endl;

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<full_object_detection>> face_boxes_test;
	load_image_dataset(images_test, face_boxes_test, faces_test_directory + "/300all_has_Part_H_testset_All.xml");
	cout << "loaded.." << endl;
	

	tinyxml2::XMLElement * dataset = doc.RootElement();
	tinyxml2::XMLElement * images = dataset->FirstChildElement("images");
	tinyxml2::XMLElement * image = images->FirstChildElement();

	FittingInst->load_blendshapes();
	FittingInst->load_3DModelPoints();
	for (int i = 0; i < images_test.size(); i++) {
		double width = images_test[i].nc();//cols = width
		double height = images_test[i].nr();
		FittingInst->calc_camera_matrix((width < height ? width : height), cv::Point2d(width / 2, height / 2));

		std::vector<full_object_detection> fods = face_boxes_test[i];

		tinyxml2::XMLElement * box = image->FirstChildElement();
		tinyxml2::XMLElement * siblingBox;
		for (auto fod: fods) {
			siblingBox = box->NextSiblingElement();
			FittingInst->fitting_shape(fod);
			vec3 vRotate;
			vec3 vTrans;
			FittingInst->getRotationAndTranslate(vRotate, vTrans);
			//if (-10.0 > vRotate.z || vRotate.z > 18.0 || -22.0 > vRotate.y || vRotate.y > 27) {//正脸筛选器
			//if (-16.0 > vRotate.z || vRotate.z > 25.0 || 6.5 > vRotate.y || vRotate.y > 40.0) {//左脸筛选器
			if (-16.0 > vRotate.z || vRotate.z > 17.0 || -40.0 > vRotate.y || vRotate.y > -2.0) {//右脸筛选器
				cout << " y: " << vRotate.y << " z: " << vRotate.z<< " deleted from ...name: " <<  image->Attribute("file")  << endl;
				cout << " box top :"<<box->Attribute("top") <<" & left :"<<box->Attribute("left")<< endl;
				image->DeleteChild(box);//若此方框不属于正、左或右脸，则从图片中删去
			}
			
			box = siblingBox;
			//cin.get();
		}
		if (!image->NoChildren()) {//若图片里还有正、左或右脸方框
			//cout << "has boxes,add to fontFaces.xml" << endl;
			fontFace.RootElement()->FirstChildElement()->InsertEndChild(image->DeepClone(&fontFace));
		}
		image = image->NextSiblingElement();
	}
	fontFace.SaveFile((faces_test_directory + "/rightFaces_H_testset.xml").c_str());
}

// 移动鼠标 选取矩形框  
void mouseClickCallback(int event,
	int x, int y, int flags, void* userdata)
{
	// 矩形数据返回  
	cv::Rect2d * pRect =
		reinterpret_cast<cv::Rect2d*>(userdata);
	// 鼠标按下操作  
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		std::cout << "LBUTTONDOWN ("
			<< x << ", " << y << ")" << std::endl;
		// 获取x，y坐标  
		pRect->x = x;
		pRect->y = y;
	}
	// 鼠标抬起操作  
	else if (event == cv::EVENT_LBUTTONUP)
	{
		std::cout << "LBUTTONUP ("
			<< x << ", " << y << ")" << std::endl;
		// 获取矩形宽高  
		pRect->width = std::abs(x - pRect->x);
		pRect->height = std::abs(y - pRect->y);
	}
}
void tracking() {
	cv::VideoCapture cap(0);
	//cv::VideoCapture cap("E:/software/Youku Files/download/test.mp4");
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return;
	}
	cv::Mat frame;

	cv::Rect2d *rect(new cv::Rect2d);
	
	//5种跟踪算法
	// "KCF"：目标人物与其他人在画面上大面积交错以后，会出现跟踪错误
	// "MIL"：速度慢，会出现跟踪错误
	//"BOOSTING"：慢，会出现跟踪错误
	//"MEDIANFLOW"：比较流畅，但是跟踪效果差，到了边界后会将方框变大
	//"TLD"：非常慢，自动调整跟踪目标的大小（框），在目标周围没有明显干扰的情况下，也会跟踪错误
	cv::Ptr<cv::Tracker> tracker;// = cv::TrackerKCF::create();
	//cv::Ptr<cv::TrackerMIL> tracker;// = cv::TrackerMIL::create();
	//cv::Ptr<cv::TrackerBoosting> tracker;// = cv::TrackerBoosting::create();
	//cv::Ptr<cv::TrackerMedianFlow> tracker;// = cv::TrackerMedianFlow::create();
	//cv::Ptr<cv::TrackerTLD> tracker;// = cv::TrackerTLD::create();


	//cap >> frame;
	//cv::resize(frame, frame, cv::Size(), 0.25, 0.25);
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	object_detector<image_scanner_type> detector_all;
	deserialize("detector_all.svm") >> detector_all;
	
	
	//cv::imshow("Tracker", frame);

	//cv::setMouseCallback("Tracker", mouseClickCallback, reinterpret_cast<void*>(rect));
	
	
	//double fps = 1000/cap.get(cv::CAP_PROP_FPS);
	bool initialized = false;
	//image_window allWin;
	//allWin.set_title("all detector");
	while (cv::waitKey(2) != 27) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		cv::flip(frame,frame, 1);

		std::vector<rectangle> detsAll;

		if (!initialized) {//若追踪器还未初始化，则初始化
			while (cv::waitKey(2) != 27) {
				cv_image<bgr_pixel> dlib_img(frame);
				detsAll = detector_all(dlib_img);//整合体检测
				if (detsAll.size() != 0) {//当检测到人脸时
					break;
				}
				cv::imshow("Tracker", frame);
				cap >> frame;
				cv::flip(frame, frame, 1);
			}

			rectangle rec = detsAll.at(0);//取第一张人脸
			rect->x = rec.left();
			rect->y = rec.top();
			rect->width = rec.width();
			rect->height = rec.height();

			tracker = cv::TrackerKCF::create();//重新初始化，这样可以避免再次更新时失败
			//tracker = cv::TrackerMedianFlow::create();
			//tracker = cv::TrackerMIL::create();
			//tracker = cv::TrackerBoosting::create();
			//tracker = cv::TrackerTLD::create();
			tracker->init(frame, *rect);
			initialized = true;
			cout << "initialized" << endl;
		}
		
		
		//追踪器更新
		if (tracker->update(frame, *rect))
			cv::rectangle(frame, *rect, cv::Scalar(255, 0, 0), 2, 1);//将追踪到的方框显示出来
		else {//追踪不到的时候再用检测器初始化追踪器：可能是侧脸和旋转脸情况
			initialized = false;
			cv::rectangle(frame, *rect, cv::Scalar(0, 0, 255), 1, 1);//将检测到的方框显示出来
			cout << "lost" << endl;
		}
		cv::imshow("Tracker", frame);

		
	}
	cap.release();
	return;
}

int main(int argc, char** argv)
{
	//detectFaces();
	//trainingDetector();
	//tt();
	//test_face_detector_L_Detector();
	//test_face_detector_A_L_I_Detector();
	test_face_detector_L_Detector_Camera();
	//test_face_detector();
	//darkenHalfImage();
	//faceRotate();
	//cameraFaceRotate();
	//divide_Rotations();
	//tracking();
}