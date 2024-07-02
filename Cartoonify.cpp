#include "stdafx.h"
#include "common.h"
#include <algorithm>
#include <cmath>
#include <opencv2/core/utils/logger.hpp>
#include <stack>

#undef min
#undef max

wchar_t* projectPath;

Mat medianFilter(const Mat& img) {
		Mat filteredImage = img.clone();

		int height = img.rows;
		int width = img.cols;
		int channels = img.channels();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int top = std::max(0, y - 3);
				int bottom = std::min(height - 1, y + 3);
				int left = std::max(0, x - 3);
				int right = std::min(width - 1, x + 3);

				Mat neighborhood = img(Range(top, bottom + 1), Range(left, right + 1));
					
					std::vector<std::pair<uchar, Point>> values;
					if (channels == 1) {
						for (int ny = 0; ny < neighborhood.rows; ny++) {
							for (int nx = 0; nx < neighborhood.cols; nx++) {
								values.push_back(
									{
										neighborhood.at<uchar>(ny, nx),
										Point(nx,ny)
									}
								);
							}
						}
					}
					else {
						for (int ny = 0; ny < neighborhood.rows; ny++) {
							for (int nx = 0; nx < neighborhood.cols; nx++) {
								Vec3b pixel = neighborhood.at<Vec3b>(ny, nx);
								uchar average = (pixel[0] + pixel[1] + pixel[2]) / 3;
								values.push_back({ 
									average, 
									Point(nx, ny)
									});
							}
						}
					}

					std::sort(values.begin(), values.end(), [](auto a, auto b) {return a.first < b.first; });
					Point median_value = values[values.size() / 2].second;

					if (channels == 1)
						filteredImage.at<uchar>(y, x) = neighborhood.at<uchar>(median_value);
					else 
						filteredImage.at<Vec3b>(y, x) = neighborhood.at<Vec3b>(median_value);
			}
		}
		imshow("filtered Image", filteredImage);
		return filteredImage;
}

Mat nonMatSuppression(const Mat& magnitude, const Mat& direction) {
	Mat suppressed = Mat::zeros(magnitude.size(), magnitude.type());

	for (int y = 1; y < magnitude.rows - 1; y++) {
		for (int x = 1; x < magnitude.cols - 1; x++) {
			float angle = direction.at<float>(y, x);

			float mag = magnitude.at<float>(y, x);
			float mag1, mag2;

			if (angle < 0) angle += 180;
			if (angle < 22.5 || angle >= 157.5) {
				mag1 = magnitude.at<float>(y, x + 1);
				mag2 = magnitude.at<float>(y, x - 1);
			}
			else if (angle < 67.5) {
				mag1 = magnitude.at<float>(y - 1, x - 1);
				mag2 = magnitude.at<float>(y + 1, x + 1);
			}
			else if (angle < 112.5) {
				mag1 = magnitude.at<float>(y - 1, x);
				mag2 = magnitude.at<float>(y + 1, x);
			}
			else {
				mag1 = magnitude.at<float>(y - 1, x + 1);
				mag2 = magnitude.at<float>(y + 1, x - 1);
			}

			if (mag >= mag1 && mag >= mag2)
				suppressed.at<float>(y, x) = mag;
			else
				suppressed.at<float>(y, x) = 0;
		}
	}

	return suppressed;
}

Mat thresholding(const Mat& magnitude, float low, float high) {
	Mat edges = Mat::zeros(magnitude.size(), CV_8U);

	for (int y = 0; y < magnitude.rows; y++) {
		for (int x = 0; x < magnitude.cols; x++) {
			float mag = magnitude.at<float>(y, x);
			if (mag >= high) {
				edges.at<uchar>(y, x) = 255;
			}

			else if (mag >= low && mag < high) {
				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						if (y + dy >= 0 && y + dy < magnitude.rows &&
							x + dx >= 0 && x + dx < magnitude.cols &&
							magnitude.at<float>(y + dy, x + dx) >= high) {
							edges.at<uchar>(y, x) = 255;
							break;
						}
					}
					if (edges.at<uchar>(y, x) == 255) {
						break;
					}
				}
			}
		}
	}

	return edges;
}

Mat applyOperations(Mat& src) {
	Mat result;

	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(src, result, element);
	return result;
}

void edgeSeparation(const Mat& edges, std::vector<std::vector<Point>>& separatedEdges) {
	Mat visited = Mat::zeros(edges.size(), CV_8UC1);

	for (int y = 0; y < edges.rows; y++) {
		for (int x = 0; x < edges.cols; x++) {
			if (edges.at<uchar>(y, x) == 255 && visited.at<uchar>(y, x) == 0) {
				std::vector<Point> edge;
				std::stack<Point> stack;
				stack.push(Point(x, y));
				while (!stack.empty()) {
					Point current = stack.top();
					stack.pop();
					if (current.x >= 0 && current.x < edges.cols &&
						current.y >= 0 && current.y < edges.rows &&
						edges.at<uchar>(current) == 255 &&
						visited.at<uchar>(current) == 0) {
						edge.push_back(current);
						visited.at<uchar>(current) = 255;
						stack.push(Point(current.x + 1, current.y));
						stack.push(Point(current.x - 1, current.y));
						stack.push(Point(current.x, current.y + 1));
						stack.push(Point(current.x, current.y - 1));
					}
				}
				separatedEdges.push_back(edge);
			}
		}
	}
}

Mat edgeFiltering(Mat& src, int min) {
	std::vector<std::vector<Point>> edges;
	edgeSeparation(src, edges);

	Mat filteredEdges = Mat::zeros(src.size(), CV_8UC1);

	for (const auto& edge : edges) {
		double area = contourArea(edge);
		if (area >= min) {
			for (const auto& point : edge) {
				int x = point.x;
				int y = point.y;
				filteredEdges.at<uchar>(y, x) = 255;
			}
		}
	}

	return filteredEdges;
}

Mat computeEdgeMask(Mat& img) {
	Mat filtered = medianFilter(img);

	Mat gradientX, gradientY;
	Sobel(filtered, gradientX, CV_32F, 1, 0, 3);
	Sobel(filtered, gradientY, CV_32F, 0, 1, 3);


	Mat magnitude, direction;
	cv::sqrt(gradientX.mul(gradientX) + gradientY.mul(gradientY), magnitude);
	phase(gradientX, gradientY, direction, true);
	direction = direction * 180 / CV_PI;

	Mat suppressed = nonMatSuppression(magnitude, direction);
	Mat edges = thresholding(suppressed, 10, 70);
	Mat morphedEdges = applyOperations(edges);
	Mat filteredEdges = edgeFiltering(morphedEdges, 10);
	imshow("Edges", edges);
	imshow("Dilated Edges", morphedEdges);
	imshow("Filtered Edges", filteredEdges);
	return filteredEdges;
}

enum ScalingType { DOWNSAMPLE, UPSAMPLE };

Mat downsample(Mat& src, int factor, ScalingType type) {
	Mat scaled;
	if (type == DOWNSAMPLE) {
		int downscaledWidth = src.cols / factor;
		int downscaledHeight = src.rows / factor;
		resize(src, scaled, Size(downscaledWidth, downscaledHeight));
	}
	else if (type == UPSAMPLE) {
		int upscaledWidth = src.cols * factor;
		int upscaledHeight = src.rows * factor;
		resize(src, scaled, Size(upscaledWidth, upscaledHeight), 0, 0, INTER_LINEAR);
	}
	return scaled;
}

Mat bilateralFiltering(Mat& src, int diameter, double sigmaColour, double sigmaSpace) {
	Mat downsampled = downsample(src, 4, DOWNSAMPLE);
	Mat filtered = Mat::zeros(downsampled.size(), downsampled.type());

	Mat spatialKernel = Mat::zeros(diameter, diameter, CV_64F);
	int radius = diameter / 2;
	double spatialCoef = -0.5 / (sigmaSpace * sigmaSpace);
	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			double distance = sqrt(x * x + y * y);
			spatialKernel.at<double>(y + radius, x + radius) = exp(distance * distance * spatialCoef);
		}
	}

	for (int y = 0; y < downsampled.rows; y++) {
		for (int x = 0; x < downsampled.cols; x++) {
			Vec3d pixelFiltered = Vec3d(0, 0, 0);
			double totalWeight = 0;

			for (int j = -radius; j <= radius; j++) {
				for (int i = -radius; i <= radius; i++) {
					int neighborY = y + j;
					int neighborX = x + i;

					if (neighborY >= 0 && neighborY < downsampled.rows && neighborX >= 0 && neighborX < downsampled.cols) {
						double spatialWeight = spatialKernel.at<double>(j + radius, i + radius);

						Vec3d diff = downsampled.at<Vec3b>(y, x) - downsampled.at<Vec3b>(neighborY, neighborX);
						double intensityWeight = exp(-0.5 * (diff.dot(diff)) / (sigmaColour * sigmaColour));

						double weight = spatialWeight * intensityWeight;

						pixelFiltered += weight * downsampled.at<Vec3b>(neighborY, neighborX);
						totalWeight += weight;
					}
				}
			}
			filtered.at<Vec3b>(y, x) = pixelFiltered / totalWeight;
		}
	}
	Mat upsampled = downsample(filtered, 4, UPSAMPLE);
	imshow("sampled", upsampled);
	return upsampled;
}

Mat quantitize(Mat& img) {
	Mat newImage = img.clone();
	for (int y = 0; y < newImage.rows; y++) {
		for (int x = 0; x < newImage.cols; x++) {
			Vec3b pixel = newImage.at<Vec3b>(y, x);
			pixel[0] = pixel[0] / 24;
			pixel[1] = pixel[1] / 24;
			pixel[2] = pixel[2] / 24;
			newImage.at<Vec3b>(y, x) = pixel * 24;
		}
	}
	imshow("Quantitized", newImage);
	return newImage;
}

void finalized(Mat edges, Mat colored) {
	Mat end = colored.clone();
	for (int y = 0; y < end.rows; y++) {
		for (int x = 0; x < end.cols; x++) {
			if (edges.at<uchar>(y, x) == 255)
				end.at<Vec3b>(y, x) = (0, 0, 0);
		}
	}
	imshow("Finalized", end);
}

int main() 
{	
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname);
		Mat imgGray = imread(fname, IMREAD_GRAYSCALE);

		Mat edgeMask = computeEdgeMask(imgGray);
		Mat newImg = bilateralFiltering(img, 9, 75, 75);
		Mat quantitized = quantitize(newImg);
		finalized(edgeMask, quantitized);
		imshow("Original", img);
		waitKey();
	
	}
	return 0;
}