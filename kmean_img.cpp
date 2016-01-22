#include <cv.h>
#include <highgui.h>

using namespace cv;

int
main (int argc, char **argv) {
	const int cluster_count = 10;  /* number of cluster */

	// (1)load a specified file as a 3-channel color image
	const char *imagename = argc > 1 ? argv[1] : "../image/boat.png";
	Mat src_img = imread(imagename);
	if(!src_img.data)
		return -1;

	// (2)reshape the image to be a 1 column matrix
	Mat points;
	src_img.convertTo(points, CV_32FC3);
	points = points.reshape(3, src_img.rows*src_img.cols);

	// (3)run k-means clustering algorithm to segment pixels in RGB color space
	Mat_<int> clusters(points.size(), CV_32SC1);
	Mat centers;
	kmeans(points, cluster_count, clusters,
	       cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 1, KMEANS_PP_CENTERS, ¢ers);

	// (4)make a each centroid represent all pixels in the cluster
	Mat dst_img(src_img.size(), src_img.type());
	MatIterator_<Vec3f> itf = centers.begin<Vec3f>();
	MatIterator_<Vec3b> itd = dst_img.begin<Vec3b>(), itd_end = dst_img.end<Vec3b>();
	for(int i=0; itd != itd_end; ++itd, ++i) {
		Vec3f color = itf[clusters(1,i)];
		(*itd)[0] = saturate_cast<uchar>(color[0]);
		(*itd)[1] = saturate_cast<uchar>(color[1]);
		(*itd)[2] = saturate_cast<uchar>(color[2]);
	}

	// (5)show source and destination image, and quit when any key pressed
	namedWindow("src_img", CV_WINDOW_AUTOSIZE);
	imshow("src_img", src_img);
	namedWindow("dst_img", CV_WINDOW_AUTOSIZE);
	imshow("dst_img", dst_img);
	waitKey(0);

	return 0;
}
