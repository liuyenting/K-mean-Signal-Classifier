#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define SHOW_PREVIEW
// #define DEBUG

const int canny_threshold = 15;
const int accumulator_threshold = 60;
const int roi_to_edge = 10;             // Blank region between ROI and the edges.

const int extract_width = 20; // Strip width during the signal extraction.

const int levels_of_signal = 5;
const cv::Scalar color_table[] =
{
	cv::Scalar(0, 0, 255),
	cv::Scalar(0,255,0),
	cv::Scalar(255,100,100),
	cv::Scalar(255,0,255),
	cv::Scalar(0,255,255)
};

const int valid_signal_duration_threshold = 10;  // Peak-to-Peak shall separate N pixels.

static cv::Point GetCenter(const std::vector<cv::Vec3f> &source, int index) {
	return cv::Point(std::round(source[index][0]), std::round(source[index][1]));
}

static int GetRadius(const std::vector<cv::Vec3f> &source, int index) {
	return std::round(source[index][2]);
}

// Step 1
static void Crop2Source(const cv::Mat &original_img,
                        cv::Mat &cropped_img, cv::Mat &signal) {
	// Generate gray scale image for blurring.
	cv::Mat gray_img;
	cv::cvtColor(original_img, gray_img, CV_BGR2GRAY);

	// Use gaussian blur to reduce the background noise.
	cv::GaussianBlur(gray_img,              // Gray-scale source image.
	                 gray_img,              // Directly blur the gray-scale image.
	                 cv::Size(27, 27),      // Gaussian kernel size, try-n-error.
	                 2, 2);                 // Kernel S.D. in X and Y direction.

	// Hold the image of detected circles.
	std::vector<cv::Vec3f> circles;

	// Start the detection.
	cv::HoughCircles(gray_img,             // Blurred gray-scale image.
	                 circles,              // Output vector.
	                 cv::HOUGH_GRADIENT,
	                 1,                    // Resolution of the internal accumulator.
	                 gray_img.rows/8,      // Minimum distance between centers.
	                 canny_threshold,
	                 accumulator_threshold,
	                 200,                  // Minimum circle radius.
	                 0);                   // Maximum circle radius, 0 as unlimited.

	// Only the largest circle is consider as our target.
	int max_radius = -1;
	cv::Point center;
	for(size_t i = 0; i < circles.size(); i++) {
		int tmp_radius = GetRadius(circles, i);
		if(tmp_radius > max_radius) {
			max_radius = tmp_radius;
			center = GetCenter(circles, i);
		}
	}

	// Crop the image to ROI.
	cv::Point delta = cv::Point(max_radius + roi_to_edge,
	                            max_radius + roi_to_edge);
	cv::Rect roi(center - delta,        // Diagnol position 1.
	             center + delta);       // Diagnol position 2.
	cropped_img = original_img.clone();
	cropped_img = cropped_img(roi);
	center = delta;            // Remove the offset, since the image is cropped.

	#ifdef SHOW_PREVIEW
	// Generate preview of the result.
	cv::Mat preview_img;
	preview_img = cropped_img.clone();

	// Circle the light source.
	cv::circle(preview_img, center, max_radius,
	           cv::Scalar(0, 0, 255),  // Color of the circle outline.
	           1);                     // Thickenss.

	// Draw the diameter.
	cv::Point y_delta = cv::Point(0, max_radius);
	cv::line(preview_img,
	         center - y_delta,
	         center + y_delta,
	         cv::Scalar(255, 0, 0),    // Color of the extracted location.
	         1);                       // Thickness.
	#endif

	// Retrieve the signal, located at the diameter.
	// signal = cropped_img.col(center.x).rowRange(center.y - max_radius,
	//                                            center.y + max_radius);
	signal = cropped_img.colRange(center.x - extract_width,
	                              center.x + extract_width).rowRange(center.y - max_radius,
	                                                                 center.y + max_radius);
	#ifdef SHOW_PREVIEW
	// Color histogram of the extracted signal.
	#endif

	#ifdef SHOW_PREVIEW
	const std::string window_title = "Cropped Image";
	cv::namedWindow(window_title, cv::WINDOW_AUTOSIZE);
	cv::imshow(window_title, preview_img);
	#endif
}

// Step 2
static void Quantized2Level(const cv::Mat &cropped_img, const int n_levels,
                            cv::Mat &quantized_img) {
	// Convert to L*a*b* color space and 32-bit float.
	cv::Mat lab_img;
	cropped_img.convertTo(lab_img, CV_32FC3);
	// cv::cvtColor(lab_img, lab_img, CV_BGR2Lab);

	/*
	   // Perform in-place unsharp masking.
	   cv::Mat blured_lab_img;
	   cv::GaussianBlur(lab_img,               // Gray-scale source image.
	                 blured_lab_img,        // Directly blur the gray-scale image.
	                 cv::Size(27, 27),      // Gaussian kernel size, try-n-error.
	                 5, 5);                 // Kernel S.D. in X and Y direction.
	   cv::addWeighted(lab_img,        // First input array.
	                2,            // Weight of the first array.
	                blured_lab_img, // Second input array.
	                -0.5,           // Weight of the second array.
	                0,              // Gamma, scalar added to each sum.
	                lab_img);
	 */

	// Reshpae the array for further K-mean processing.
	// (M, N, n-channels) -> (MxN, n-channels)
	cv::Size img_size = cropped_img.size();
	int n_pixels = img_size.width * img_size.height;
	lab_img = lab_img.reshape(0, n_pixels);

	// Apply K-mean.
	cv::Mat_<int> indices(lab_img.size(), CV_32SC1);
	#ifndef DEBUG
	cv::kmeans(lab_img,  // Sample array, 1 row per sample.
	           n_levels, // Number of clusters to split the set by.
	           indices,  // Integer artray that stores the cluster indices for every sample.
	           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
	           3,   // Number of times the algorithm executed with different init labels.
	           cv::KMEANS_PP_CENTERS);
	#else
	cv::Mat centers;
	cv::kmeans(lab_img,  // Sample array, 1 row per sample.
	           n_levels, // Number of clusters to split the set by.
	           indices,  // Integer artray that stores the cluster indices for every sample.
	           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
	           3,   // Number of times the algorithm executed with different init labels.
	           cv::KMEANS_PP_CENTERS,
	           centers);
	std::cerr << "centers = " << std::endl << " " << centers << std::endl << std::endl;
	#endif

	// Fit the result, change the pixel values to centers of the clusters.
	quantized_img = cv::Mat(lab_img.size(), CV_8UC1);
	cv::MatIterator_<uchar> pixel = quantized_img.begin<uchar>();
	cv::MatIterator_<uchar> pixel_end = quantized_img.end<uchar>();
	for(int i = 0; pixel != pixel_end; i++, ++pixel) {
		// Offset the levels by 1 to avoid divide-by-0.
		int cluster_index = (levels_of_signal+1) - indices(1,i);
		// Evenly distributed in the gray scale.
		*pixel = 255 / cluster_index;
	}

	// Reshape back to the target image size (M, N, n-channels).
	quantized_img = quantized_img.reshape(0, img_size.height);

	// Average the rows to acquire a single line.
	cv::reduce(quantized_img, quantized_img,
	           1,                    // Reduce to a single column.
	           CV_REDUCE_AVG);

	#ifdef SHOW_PREVIEW
	const std::string window_title = "Quantized Signal";
	cv::namedWindow(window_title, cv::WINDOW_AUTOSIZE);
	cv::imshow(window_title, quantized_img);
	#endif
}

// Step 3
static void IdentifyPosition(const cv::Mat &quantized_signal, const int threshold,
                             std::vector<int> &measure_pos) {
	// Wipe the vector.
	measure_pos.clear();

	// Find all the local maxima.
	// Since the data is quantized, only the difference is required,
	//  and ignore all the signals that are too close to each other.
	int n_rows = quantized_signal.size().height;
	uchar tmp_val = 255, curr_val;
	for(int i = 0, distance = 0; i < n_rows; i++, distance++) {
		curr_val = quantized_signal.at<uchar>(i,1);

		#ifdef DEBUG
		std::cerr << "prev value = " << (int)tmp_val << std::endl;
		std::cerr << "curr value = " << (int)curr_val << std::endl;
		std::cerr << "distance   = " << distance << std::endl;
		#endif

		if(curr_val != tmp_val) {
			tmp_val = curr_val;
			if(distance > threshold) {
				std::cerr << "...different" << std::endl;
			} else {
				if(!measure_pos.empty())
					measure_pos.pop_back();
			}
			distance = 0;
			measure_pos.push_back(i);
		}
	}

	#ifdef SHOW_PREVIEW
	cv::Mat preview;

	// Extend the column to visible size.
	cv::repeat(quantized_signal, 1,
	           2*extract_width + 1,
	           preview);

	// Dump the vector back to the profile matrix.
	int x_pos = extract_width + 1, y_pos;
	for(unsigned int i = 1; i < measure_pos.size(); i++) {
		y_pos = (measure_pos[i-1] + measure_pos[i]) / 2;
		cv::circle(preview,
		           cv::Point(x_pos, y_pos),
		           2, cv::Scalar(255), -1);
	}

	cv::namedWindow("Scharr Result", cv::WINDOW_AUTOSIZE);
	cv::imshow("Scharr Result", preview);
	#endif
}

// Step 4
static void ExtractSignal(const cv::Mat &quantized_signal, const std::vector<int> pos,
                          std::vector<int> &labeled_signal) {

}

int main(int argc, char **argv) {
	if(argc < 2) {
		std::cerr << "No input image specified.\n" << std::endl;
		std::cerr << "usage: " << argv[0] << " [path to image]" << std::endl;
		return -1;
	}

	// Read the image.
	cv::Mat original_img = cv::imread(argv[1], 1);
	if(original_img.empty()) {
		std::cerr << "Invlaid image.\n";
		return -1;
	}

	cv::Mat cropped_img; // Cropped to the targeted light source.
	cv::Mat raw_signal;   // The desired signal pattern extracted from the diameter.
	Crop2Source(original_img, cropped_img, raw_signal);

	cv::Mat quantized_signal; // Color quantized signal.
	Quantized2Level(raw_signal, levels_of_signal, quantized_signal);

	std::vector<int> measure_pos; // Locations to perform the measurement.
	IdentifyPosition(quantized_signal, valid_signal_duration_threshold, measure_pos);

	#ifdef SHOW_PREVIEW
	// Wait for key press.
	cv::waitKey(0);

	// Close all the opened windows.
	cv::destroyAllWindows();
	#endif

	return 0;
}
