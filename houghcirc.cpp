#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/mat.hpp>

// windows and trackbars name
const std::string windowName = "NLC Hough Circle Detection";
const std::string profileWindowName = "Line Profile";
const std::string clusteredWindowName = "Line Profile (K-mean)";
const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

const int sigLevelCount = 5;

namespace cv
{
void Kmean(const Mat& src_prof) {
	Mat points;
	src_prof.convertTo(points, CV_32FC3);

	// reshape the image to be a 1 column matrix
	points = points.reshape(3, src_prof.rows * src_prof.cols);

	// run k-means clustering algorithm
	Mat_<int> clusters(points.size(), CV_32SC1);
	Mat centers;
	kmeans(points, sigLevelCount, clusters,
	       TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
	       3, KMEANS_PP_CENTERS, centers);

	// mae each centroid represent all pixels in the cluster
	Mat dst_prof(src_prof.size(), src_prof.type());
	MatIterator_<Vec3f> itf = centers.begin<Vec3f>();
	MatIterator_<Vec3b> itd = dst_prof.begin<Vec3b>(), itd_end = dst_prof.end<Vec3b>();
	for(auto i = 0; itd != itd_end; ++itd, ++i) {
		Vec3f color = itf[clusters(1, i)];
		for(int j = 0; j < 3; j++)
			(*itd)[j] = saturate_cast<uchar>(color[j]);
	}

	// split the rgb channels
	std::vector<Mat> rgbChannels(3);
	split(dst_prof, rgbChannels);

	// find the min and max
	double min[3], max[3];
	for(int i = 0; i < 3; i++)
		minMaxLoc(rgbChannels[i], &min[i], &max[i]);

	for(int i = 0; i < 3; i++)
		std::cout << "channel " << i << "\nmin = " << min[i] << ", max = " << max[i] << std::endl;

	// plot the line profile
	std::vector<Scalar> color = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255)};
	int max_scale = 256;
	int n_rows = dst_prof.rows, n_channels = dst_prof.channels();
	Mat histImg = Mat::zeros(n_rows, max_scale*n_channels, CV_8UC3);
	for(int j = 0; j < n_rows; j++) {
		Vec3b pixel = dst_prof.at<Vec3b>(j, 0);
		for(int c = 0; c < n_channels; c++) {
			double intensity = (double)(pixel[c]-min[c])/max[c] * max_scale;
			Point dot = Point(intensity + max_scale*c, j);
			circle(histImg, dot, 1, color[c], -1, 8, 0);
		}
	}
	imshow(clusteredWindowName, histImg);
}

void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold) {
	// will hold the results of the detection
	std::vector<Vec3f> circles;
	// runs the actual detection
	HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/8, cannyThreshold, accumulatorThreshold, 200, 0 );

	// clone the colour, input image for displaying purposes
	Mat display = src_display.clone();

	// find the circle with the max radius
	int radius = cvRound(circles[0][2]);
	Point center = Point(cvRound(circles[0][0]), cvRound(circles[0][1]));
	for(size_t i = 1; i < circles.size(); i++) {
		int tmp = cvRound(circles[i][2]);
		if(tmp > radius) {
			radius = tmp;
			center = Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
		}
	}

	// crop the image to roi
	int blank = 10;
	Point size = Point(radius+blank, radius+blank);
	Rect roi(center-size, center+size);
	display = display(roi);
	center = size;

	// circle center
	// circle( display, center, 3, Scalar(0,255,0), -1, 8, 0 );

	// circle outline
	circle( display, center, radius, Scalar(0,0,255), 1, 8, 0 );

	// retrieve the line
	Point yRadius(0, radius);
	Mat profile = display.col(center.x).rowRange((center-yRadius).y, (center+yRadius).y);

	// split the rgb channels
	std::vector<Mat> rgbChannels(3);
	split(profile, rgbChannels);

	// find the min and max
	double min[3], max[3];
	for(int i = 0; i < 3; i++)
		minMaxLoc(rgbChannels[i], &min[i], &max[i]);

	for(int i = 0; i < 3; i++)
		std::cout << "channel " << i << "\nmin = " << min[i] << ", max = " << max[i] << std::endl;

	// plot the line profile
	std::vector<Scalar> color = {Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255)};
	int max_scale = 256;
	int n_rows = profile.rows, n_channels = profile.channels();
	Mat histImg = Mat::zeros(n_rows, max_scale*n_channels, CV_8UC3);
	for(int j = 0; j < n_rows; j++) {
		Vec3b pixel = profile.at<Vec3b>(j, 0);
		for(int c = 0; c < n_channels; c++) {
			double intensity = (double)(pixel[c]-min[c])/max[c] * max_scale;
			Point dot = Point(intensity + max_scale*c, j);
			circle(histImg, dot, 1, color[c], -1, 8, 0);
		}
	}
	imshow(profileWindowName, histImg);

	// center diameter
	line(display, center-yRadius, center+yRadius, Scalar(255,0,0), 1, 0);

	// shows the results
	imshow(windowName, display);

	Kmean(profile);
}
}

int main(int argc, char** argv) {
	cv::Mat src, src_gray;

	if(argc < 2) {
		std::cerr << "No input image specified\n";
		std::cout << usage;
		return -1;
	}

	// Read the image
	src = cv::imread(argv[1], 1);

	if(src.empty()) {
		std::cerr << "Invalid input image\n";
		std::cout << usage;
		return -1;
	}

	// declare and initialize both parameters that are subjects to change
	int cannyThreshold = 15;
	int accumulatorThreshold = 60;

	// Convert it to gray
	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, cv::Size(27, 27), 2, 2);

	// create the main window, and attach the trackbars
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(profileWindowName, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(clusteredWindowName, cv::WINDOW_AUTOSIZE);

	// runs the detection, and update the display
	HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold);

	cv::waitKey(0);

	return 0;
}
