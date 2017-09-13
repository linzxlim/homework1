#include "SubImageMatch.h"

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	//realization
	return 0;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	return 0;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	return 0;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	return 0;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	return 0;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	return 0;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	return 0;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	return 0;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	return 0;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	return 0;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	return 0;
}
