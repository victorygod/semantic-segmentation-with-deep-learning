#include <iostream>  
#include<../Sparse>
#include<../Dense>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/mat.hpp>
#include<opencv2/core/eigen.hpp>


//还有这种操作？

using namespace std;
using namespace cv;
using namespace Eigen;

#define e 2.7182818
#define max(a, b) (a) < (b) ? (b): (a)
#define min(a, b) (a) < (b) ? (a): (b)
typedef Eigen::Triplet<double> T;

int getSlope(Point2f a, Point2f b) {//获得斜率，其实并没有用上
	float k = (b.y - a.y) / (b.x - a.x);
}

void getWeight(int i_row, int i_col, int j_row, int j_col, float step, Eigen::SparseMatrix<float> &WIC) {//获得两点之间的权重，额，不知道怎么写，还没写好
	float k = (j_row - i_row) / (j_col - i_col);
	int rs = min(i_row, j_row);
	int re = max(i_row, j_row);
	int cs = min(i_col, j_col);
	int ce = max(i_col, j_col);
	for (float m = cs; m < ce; m += 4*step) {
		int i = floorf(m); 
		int j = round(m*k);
		}
		
	}




void filter(const Mat &src, Mat &dst, Mat& filter, int filter_size) {//废了，别管
	dst = src.clone();
	int anchor = floor(filter_size / 2);
	int r = src.rows;
	int c = src.cols;

	for (int i = 0; i < r; i++) {
		const float* colData = src.ptr<float>(i);
		float* res = dst.ptr<float>(i);//response
		if (i < anchor) {
			for (int j = 0; j <c; j++) {
				for (int fr = anchor + 1; fr < filter_size; fr++) {
					float* colFilter = filter.ptr<float>(fr);
					int st, end;
					if (j < anchor) {
						st = -j;
						end = anchor + 1;
					}
					if (j >= anchor&&j<=(c-anchor)) {
						st = -anchor;
						end = anchor + 1;
					}
					if (j > (c - anchor)&&j<c) {
						st = -anchor;
						end = c - j;
					}
					for (int fc = st; fc < end; fc++) {
						res[j] += colData[j + fc] * colFilter[anchor + fc];
					}
				}
			}

		}

		if (i >= anchor&&i <= r - anchor) {
			for (int j = 0; j < anchor; j++) {
				for (int fr = 0; fr < filter_size; fr++) {
					float* colFilter = filter.ptr<float>(fr);
					for(int fc = -j;fc<anchor+1;fc++){
						res[j] += colData[j + fc] * colFilter[anchor + fc];
					}
				}

			}
			for (int j = c-anchor; j < c; j++) {
				for (int fr = 0; fr < filter_size; fr++) {
					float* colFilter = filter.ptr<float>(fr);
					for(int fc = -anchor;fc<c-j;fc++){
						res[j] += colData[j + fc] * colFilter[anchor + fc];
					}
				}

			}
		}

		if (i > (r - anchor) && i < r) {
			
			for (int j = 0; j < anchor; j++) {
				for (int fr = 0; fr < anchor+1; fr++) {
					float* colFilter = filter.ptr<float>(fr);
					for (int fc = -j; fc<anchor + 1; fc++) {
						res[j] += colData[j + fc] * colFilter[anchor + fc];
					}
				}
			}

			for (int j = anchor; j < c-anchor; j++) {
				for (int fr = 0; fr < anchor + 1; fr++) {
					float* colFilter = filter.ptr<float>(fr);
					for (int fc = -j; fc<anchor + 1; fc++) {
						res[j] += colData[j + fc] * colFilter[anchor + fc];
					}
				}
			}


		}
		

	}

	for (int i = anchor; i < r-anchor; i++) {
		const float* colData = src.ptr<float>(i);
		float* res = dst.ptr<float>(i);//response
		for (int j = anchor; j < c-anchor; j++) {
			for (int fr = -anchor; fr < anchor+1; fr++) {
				float* colFilter = filter.ptr<float>(fr);
				for (int fc = -anchor; fc < anchor+1; fc++) {
					res[j] += colData[j + fc] * colFilter[anchor + fc];
				}
			}
		}
	}

}

void rotate(const Mat &src1, const Mat &src2, Mat &dst1, Mat &dst2, int angle, Point2f center) {//给滤波器做旋转的，用在contourMat函数里头，参数src1和src2是一阶/二阶拉普拉斯高斯滤波器，两个一起旋转同样角度
	Mat rot = getRotationMatrix2D(center, angle, 1);
	warpAffine(src1, dst1, rot, src1.size());
	warpAffine(src2, dst2, rot, src2.size());
}

void contourMat(const Mat &src, Mat &dst, int filter_size =9, float theta0 = 0.8, float theta1 = 2.4){//生成轮廓图的函数
	Point2f center(floor((filter_size) / 2), floor((filter_size) / 2));

	/*imshow("src", src);
	waitKey();*/

	Mat gk_y = getGaussianKernel(filter_size, theta0, CV_32FC1);//gaussian_kernel
	Mat gk_x = getGaussianKernel(filter_size, theta1, CV_32FC1);

	Mat sobel_y(gk_y.size(), CV_32FC1);//Laplacian of g_k
	Sobel(gk_y, sobel_y, -1, 0, 1);
	

	Mat laplacian2_y(gk_y.size(), CV_32FC1);//Second Order Laplacian of g_k
	Laplacian(gk_y, laplacian2_y, CV_32FC1);
	//Laplacian(laplacian2_y, laplacian2_y, CV_32FC1);

	Mat LoG_0 = sobel_y*gk_x.t();//= laplacian_y*gk_x.t();
	Mat L2oG_0 = laplacian2_y*gk_x.t();

	/*getDerivKernels(LoG_0, L3oG_0, 2, 0, 1, true);
	L2oG_0 = LoG_0*L3oG_0.t();
	getDerivKernels(LoG_0, L3oG_0, 1, 0, 1, true);
	LoG_0 = LoG_0*L3oG_0.t();*/
	///*0degree*/
	//Mat LoG_0 = laplacian_kc*gk_first_order.t(); //LoG operator
	//Mat L2oG_0 = laplacian2_kc*gk_first_order.t();//LoG operator
	//cout << L2oG_0 << endl;
	/*normalize(LoG_0, LoG_0, 1.0, 0.0, NORM_L1);
	cout << sum(LoG_0) << endl;
	normalize(L2oG_0, L2oG_0, 1.0, 0.0, NORM_L1);*/

	Mat rot_0 = getRotationMatrix2D(center, 0, 3);
	warpAffine(LoG_0, LoG_0, rot_0, LoG_0.size());
	warpAffine(L2oG_0, L2oG_0, rot_0, LoG_0.size());

	for (int i = 0; i < 180; i += 30) {
		Mat LoG_00(LoG_0.size(), CV_32FC1);
		Mat L2oG_00(LoG_0.size(), CV_32FC1);
		rotate(LoG_0, L2oG_0, LoG_00, L2oG_00, i, center);
		Mat temp(dst.rows, dst.cols, CV_32FC1);
		filter2D(src, temp, CV_32FC1, LoG_00);
		dst += temp.mul(temp);
		//normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);
		filter2D(src, temp, CV_32FC1, L2oG_00);
		dst += temp.mul(temp);
		//normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);
		/*for (int m = 0; m < dst.rows;  m++) {
			for (int n = 0; n < dst.cols; n++) {
				cout << dst.at<float>(m, n) << endl;
				dst.at<float>(m, n) = -(expf((dst.at<float>(m, n))/20));
				cout << dst.at<float>(m, n) << endl;
			}
		}
		Mat p = 1 - dst;
		normalize(p, p, 1.0, 0.0, NORM_MINMAX);
		imshow("p", p);
		waitKey(600);*/
		cout << "filtering..." << i << endl;
		
	}

	/*30degree*/
	Mat rot_30 = getRotationMatrix2D(center, 30, 1);
	Mat LoG_30(LoG_0.size(), CV_32FC1);
	Mat L2oG_30(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_30, rot_30, LoG_30.size());
	warpAffine(L2oG_0, L2oG_30, rot_30, LoG_30.size());

	/*60degree*/
	Mat rot_60 = getRotationMatrix2D(center, 60, 1);
	Mat LoG_60(LoG_0.size(), CV_32FC1);
	Mat L2oG_60(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_60, rot_60, LoG_60.size());
	warpAffine(L2oG_0, L2oG_60, rot_60, LoG_60.size());

	/*90degree*/
	Mat rot_90 = getRotationMatrix2D(center, 90, 1);
	Mat LoG_90(LoG_0.size(), CV_32FC1);
	Mat L2oG_90(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_90, rot_90, LoG_90.size());
	warpAffine(L2oG_0, L2oG_90, rot_90, LoG_90.size());


	/*120degree*/
	Mat rot_120 = getRotationMatrix2D(center, 120, 1);
	Mat LoG_120(LoG_0.size(), CV_32FC1);
	Mat L2oG_120(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_120, rot_120, LoG_120.size());
	warpAffine(L2oG_0, L2oG_120, rot_120, LoG_120.size());

	/*150degree*/
	Mat rot_150 = getRotationMatrix2D(center, 150, 1);
	Mat LoG_150(LoG_0.size(), CV_32FC1);
	Mat L2oG_150(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_150, rot_150, LoG_150.size());
	warpAffine(L2oG_0, L2oG_150, rot_150, LoG_150.size());

	/*180degree*/
	Mat rot_180 = getRotationMatrix2D(center, 180, 1);
	Mat LoG_180(LoG_0.size(), CV_32FC1);
	Mat L2oG_180(LoG_0.size(), CV_32FC1);
	warpAffine(LoG_0, LoG_180, rot_180, LoG_180.size());
	warpAffine(L2oG_0, L2oG_180, rot_180, LoG_180.size());

	//normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);
	//normalize(u, u, 1.0, 0.0, NORM_MINMAX);
	//cout << src << endl;
	//Mat temp(dst.rows, dst.cols, CV_32FC1);
	//filter2D(src, temp, CV_32FC1, LoG_90);
	//dst = temp.mul(temp);
	normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);
	//imshow("filtered_1", temp);
	//double min0, max0;
	//minMaxLoc(temp, &min0, &max0);
	//cout << "min: " << min0 << ", max: " << max0 << endl;
	////cout << temp.mul(temp) << endl;
	////normalize(src, src, 1.0, 0.0, NORM_MINMAX);
	//filter2D(src, temp, CV_32FC1, L2oG_90);

	////Laplacian(src, temp, CV_32FC1, 1);
	////cout << temp << endl;
	//dst += temp.mul(temp);
	////cout << dst << endl;
	///*normalize(src, src, 1.0, 0.0, NORM_MINMAX);
	//dst = dst - src;
	//normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);*/
	//double min, max;
	//minMaxLoc(dst, &min, &max);
	//cout << "min: " << min << ", max: " << max << endl;

	for (int m = 0; m < dst.rows;  m++) {
	for (int n = 0; n < dst.cols; n++) {
	//cout << dst.at<float>(m, n) << endl;
	dst.at<float>(m, n) = (expf((dst.at<float>(m, n))/-0.4));
	//cout << dst.at<float>(m, n) << endl;
	}
	}
	dst = 1 - dst;

	imshow("filtered_2", dst);
	waitKey();

}

void preProcessing(Mat& src, Mat& dst) {//其实也还没用到的图像预处理函数
	vector<Mat> GRBChannels;
	split(src, GRBChannels);

	vector<Mat> YUVChannels;
	Mat yuv_src;
	cvtColor(src, yuv_src, CV_BGR2HSV);
}


int main() {
	//Mat srcimg = imread("741681775.jpg");
	//Mat srcimg = imread("363986180.jpg");
	Mat srcimg = imread("641059184.jpg");//读入图像
	//Mat srcimg = imread("735436.png");
	Mat dstimg = srcimg.clone();
	Mat testimg = srcimg.clone();
	vector<Mat> testChan;
	split(testimg, testChan);
	Range r = Range(800, 1200);
	Range c = Range(800, 1200);
	Mat u = dstimg;// (c, r);//u and dstimg share a same data area
	int newHeight = 500;
	int newWidth = u.cols*newHeight / u.rows;
	resize(u, u, Size(newWidth, newHeight));//缩小一下
	u.convertTo(u, CV_32FC1);
	Mat YUV;
	Mat gray;
	cvtColor(u,YUV,CV_BGR2HSV);//转换成HSV三分量图，虽然目标图像的名字叫YUV但是其实是张HSV图……
	cvtColor(u, gray, CV_BGR2GRAY);
	/*cout << u << endl;
	cout << endl;
	cout << YUV << endl;*/
	//cvtColor(YUV,YUV,COLOR_YUV2BGR);
	vector<Mat> channels;
	split(YUV, channels);//HSV图分割成三个channel
	//cout << channels[2] << endl;
	channels[2].convertTo(channels[2], CV_32FC1);//取第三个channel，亮度值，转换为32FC1类型的数据
	normalize(channels[2], channels[2], 1.0, 0.0, NORM_MINMAX);//均一化一下
	testChan[2].convertTo(testChan[2], CV_32FC1);
	//gray.convertTo(channels[2], CV_32FC1);
	Mat dst(u.rows, u.cols, CV_32FC1);
	contourMat(channels[2], dst);//得轮廓图dst

	const int pix = u.cols*u.rows;//总像素个数
	Eigen::SparseMatrix<float> sparse_weight(pix, pix);//稀疏权重图W
	Eigen::SparseMatrix<float> D(pix, pix);//计算过程中用到的对角矩阵D，对角元素(i,i)为点i到其他所有点的距离之和
	Eigen::SparseMatrix<float> L(pix, pix);//计算过程中用到的对称矩阵L = D-W
	Eigen::SparseMatrix<float> DD(pix, pix);//计算过程中用到的对称矩阵DD = D^(1/2)
	Eigen::VectorXf vec(pix);
	Eigen::VectorXf rosslyn(pix);
	for (int i = 0; i < pix; i++) {
		vec(i) = 1;
	}
	//Eigen::Matrix<double, Dynamic, Dynamic> sparse_weight(pix, pix);
	//Eigen::Matrix<double, Dynamic, Dynamic> D(pix, pix);
	sparse_weight.reserve(VectorXi::Constant(pix, 60));
	D.reserve(VectorXi::Constant(pix, 1));//稀疏矩阵每列保留存储空间
	L.reserve(VectorXi::Constant(pix, 65));
	DD.reserve(VectorXi::Constant(pix, 1));
	cout << sparse_weight.cols() << ", " << sparse_weight.rows() << endl;
	int uc = channels[2].cols;
	int ur = channels[2].rows;
	int i = 0;
	double omega0 = 0;
	while (i < pix) {
		int i_row = round(i / uc);
		int i_col = i % uc;
		//int j = i-36;
		float ij = 0;
		for (int k = 0; k < 90; k += uc) {
			for (int j = i - 90 + k; j < i + 90 + k; j += 4) {//额大概就是取i附近的90个点吧……计算一下亮度差值
				if (j < 0) {
					continue;
				}
				if (j >= pix) {
					break;
				}

				int j_row = round(j / uc);
				int j_col = (j % uc);

				float br = abs(((channels[2]).at<float>(j_row, j_col) - (channels[2]).at<float>(i_row, i_col)));//两点间亮度值差
				//float w = expf(-br / 10)*expf(-dis / 10);
				float w = br;
				//cout << "i: " << i << ", j: " << j << endl;
				sparse_weight.insert(i, j) = w;
				ij += w;
			}
		
		}
		//sparse_weight.insert(i, i) = -ij;
		D.insert(i, i) = ij;
		DD.insert(i, i) = powf(ij, -0.50);
		//L.insert(i, i) = ij;
		//D(i, i) = ij;
		omega0 += ij;
		//cout << omega0 << endl;
		i ++;
	}
	//cout << omega0 << endl;//overflow!!!
	L = D - sparse_weight;
	Eigen::SparseMatrix<float> A = DD*L*DD;
	cout << "A.rows: " << A.rows() << endl;
	cout << "A.cols: " << A.cols() << endl;
	cout << "vec.rows: " << vec.rows() << endl;
	cout << "vec.cols: " << vec.cols() << endl;
	rosslyn = A*vec;

	float betaone = rosslyn.blueNorm();

	cout << betaone << endl;

	Eigen::VectorXf q2 = rosslyn / betaone;

	Eigen::VectorXf y = D.cwiseSqrt()*q2;

	cout << y << endl;//上面是用数值方式计算出来的特征向量但是我忘记y和q2哪个是特征向量了嘿嘿嘿

	Map<Matrix<float, Dynamic, Dynamic>> mq2(y.data(), u.cols, u.rows);

	Matrix<float, Dynamic, Dynamic, RowMajor> rmq2= mq2.transpose();

	Mat fin(rmq2.rows(), rmq2.cols(), CV_32FC1, rmq2.data());//以上三行是特征向量重排列

	//threshold(fin, fin, 0.0001, 0.5, THRESH_BINARY);

	normalize(fin, fin, 1.0, 0.0, NORM_MINMAX);


	imshow("filtered", fin);
	//imwrite("output.png", u);
	waitKey();

	

	
	//cout << dist << endl;

	//这后面的其实也不用看

	cout << "weight matrix generated" << endl;
 	Mat gaussian_kernel = getGaussianKernel(9,0.8,CV_32FC1);
	Mat laplacian_kc(gaussian_kernel.size(),CV_32FC1);
	Mat laplacian2_kc(gaussian_kernel.size(), CV_32FC1);
	Laplacian(gaussian_kernel, laplacian_kc, CV_32FC1);
	Laplacian(laplacian_kc, laplacian2_kc, CV_32FC1);
	//cout << laplacian_kc << endl;
	//Mat LoG = gaussian_kernel*laplacian_kc.t();
	//Mat L2oG = gaussian_kernel*laplacian2_kc.t();
	Mat LoG = laplacian_kc*gaussian_kernel.t();
	Mat L2oG = laplacian2_kc*gaussian_kernel.t();
	cout << "meaningless words-----------------------------------------" << endl;
	//cout << LoG << endl;
	Point2f center(floor((LoG.rows)/2), floor((LoG.cols) / 2));

	//Mat rot_45 = getAffineTransform(src, dst);

	Mat rot_45 = getRotationMatrix2D(center, -30, 1);
	Mat LoG_45(LoG.size(), CV_32FC1);
	warpAffine(LoG, LoG_45, rot_45, LoG_45.size());

	filter2D(channels[2], u, CV_32FC1, LoG);
	normalize(u, u, 1.0, 0.0, NORM_MINMAX);
	//cout << u << endl;
	imshow("filtered", u);
	//imwrite("output.png", u);
	waitKey();
	//cout << dstimg(r, c) << endl;
	/*Point p = srcimg.at<uchar>(100, 100,2);
	Point q = srcimg.at<uchar>(500, 500,2);
	cout << "p: " << p << ", q: " << q << endl;*/
	//imgROI = srcimg(Rect(p,q));
	/*namedWindow("cat0",1);   
	namedWindow("cat1",1);
	namedWindow("cat2",1);
	imshow("cat0", channels[0]);  
	imshow("cat1", channels[1]);
	imshow("cat2", channels[2]);
	waitKey(6000);*/
	//Vec3b a = dstimg.at<Vec3b>(40, 42, 2);//here a random error occur
	//cout << a << endl;	

	system("pause");

}

