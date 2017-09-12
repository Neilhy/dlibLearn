#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/svm_threaded.h>
#include<dlib/image_processing.h>
#include<dlib/data_io.h>
#include<dlib/gui_widgets.h>
#include<iostream>
#include<fstream>
#include<string>

using namespace dlib;
using namespace std;
int test_face_detector(int argc, char ** argv) {
	if (argc != 3)
	{
		cout << "Give the images directory." << endl;
		cout << "./tester ./svm ../../../dlibTrain/xml" << endl;
		cout << endl;
		return 0;
	}
	const string faces_svm = argv[1];
	const string faces_test_xml = argv[2];

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
	object_detector<image_scanner_type> detector;
	deserialize(faces_svm) >> detector;

	frontal_face_detector detectorStandard = get_frontal_face_detector();

	dlib::array<array2d<unsigned char>> images_test;
	std::vector<std::vector<rectangle>> face_boxes_test;
	load_image_dataset(images_test, face_boxes_test, faces_test_xml);

	//upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
	cout << "num testing images:  " << images_test.size() << endl;
	cout << "standard testing results:  " << test_object_detection_function(detectorStandard, images_test, face_boxes_test) << endl;//testing
	cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;//testing
	return 0;
}
int t(int argc,char ** argv) {
	try
	{
		if(argc != 3)
		{
			cout<<"Give the images directory."<<endl;
			cout<<"./trainer ../../../dlibTrain/...xml ../../../dlibTrain/...xml"<<endl;
			cout<<endl;
			return 0;
		}
		const string faces_train_xml = argv[1];
		const string faces_test_xml = argv[2];

		dlib::array<array2d<unsigned char>> images_train,images_test;
		std::vector<std::vector<rectangle>> face_boxes_train,face_boxes_test;
		cout << "Before loading images..." << endl;
		load_image_dataset(images_train, face_boxes_train, faces_train_xml);
		cout << "After loading images..." << endl;

		//upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);
		cout << "num training images: " << images_train.size() << endl;

		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;// 5/6 rate down size
		image_scanner_type scanner;
		scanner.set_nuclear_norm_regularization_strength(1.0);		
		scanner.set_detection_window_size(80, 80);//minimize window
		scanner.set_cell_size(8);
		scanner.set_padding(0);

		structural_object_detection_trainer<image_scanner_type> trainer(scanner);

		trainer.set_num_threads(12);
		trainer.set_c(700);
		trainer.be_verbose();
		trainer.set_epsilon(0.05);//precision
		trainer.set_loss_per_missed_target(1);

		cout << "Before training images..." << endl;
		object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);
		detector = threshold_filter_singular_values(detector, 0.15);//singular value threshold: 0.15
		serialize("fontface_detector_A_H_L_I_Merge.svm") << detector;

		cout << "training results: " << test_object_detection_function(detector, images_train, face_boxes_train) << endl;//testing  precision, recall, average precision
		
		load_image_dataset(images_test, face_boxes_test, faces_test_xml);
		//upsample_image_dataset<pyramid_down<2> >(images_test, face_boxes_test);
		cout<<"num testing images: "<< images_test.size() <<endl;
		cout<<"testing results: "<< test_object_detection_function(detector,images_test, face_boxes_test) <<endl;		
		
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
	return 0;
}
/*
int main(int argc, char** argv)
{
	return t(argc,argv);
}
*/