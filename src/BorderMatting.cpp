#include "BorderMatting.h"
using namespace cv;
#define PI  (3.14159)


BorderMatting::BorderMatting(const Mat& originImage, const Mat& mask)
{
	Initialize(originImage, mask);
}


void BorderMatting::computeNearestPoint()
{
	for (int i = 0; i < Image.rows; i++) {
		for (int j = 0; j < Image.cols; j++)
		{
			// ��������������ǰ��
			if (Edge.at<uchar>(i, j) == 0 && Mask.at<uchar>(i, j) == 1)
			{
				point p(j, i);
				double mindis = INFINITY;
				int max = -1;
				int id = 0;
				for (int k = 0; k < contourVector.size(); k++) {
					double dis = p.distance(contourVector[k].pointInfo);
					if(dis < mindis){
						mindis = dis;
						id = k;
					}
				}
				if (mindis > 3) {
					continue;
				}
				else {
					p.dis = mindis;
					contourVector[id].neighbor.push_back(p);
				}
			}
		}
	}
}

// ������������¼
void BorderMatting::drawContour()
{
	for (int i = 0; i < Image.rows; i++) {
		for (int j = 0; j < Image.cols; j++)
		{
			if (Edge.at<uchar>(i, j))
			{
				contourVector.push_back(point(j, i));
			}
		}
	}
}


void BorderMatting::Initialize(const Mat& originImage, const Mat& mask, int threadshold_1, int threadshold_2)
{
	mask.copyTo(this->Mask);
	Mask = Mask & 1;
	originImage.copyTo(this->Image);
	Canny(Mask, Edge, threadshold_1, threadshold_2);   //��Ե��ȡ
	drawContour(); // ����Edge����ͼ������
	computeNearestPoint();
	haveEdge = true;
}


int BorderMatting::computeEdgeDistance(point p)
{
	for (int i = 0; i < contourVector.size(); i++)
	{
		if (p.distance(contourVector[i].pointInfo) < edgeRadius) // 3
			return p.distance(contourVector[i].pointInfo);
	}
	return -1;
}





void BorderMatting::computeMeanVariance(point p, Info &res)
{
	const int halfL = 20; // �����е�L=41
	Vec3b backMean, FrontMean;
	double backVariance = 0, frontVariance = 0;
	int frontCounter = 0, backCounter = 0;
	int x = (p.x - halfL < 0) ? 0 : p.x - halfL;
	int width = (x + 2 * halfL + 1 <= Image.cols) ? halfL * 2 + 1 : Image.cols - x;
	int y = (p.y - halfL < 0) ? 0 : p.y - halfL;
	int height = (y + 2 * halfL + 1 <= Image.rows) ? halfL * 2 + 1 : Image.rows - y;
	Mat neiborPixels = Image(Rect(x,y, width, height));
	for (int i = 0; i < neiborPixels.rows; i++) {
		for (int j = 0; j < neiborPixels.cols; j++)
		{
			Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
			if (Edge.at<uchar>(y + i, x + j) == 1)
			{
				FrontMean += pixelColor;
				frontCounter++;
			}
			else
			{
				backMean += pixelColor;;
				backCounter++;
			}
		}
	}

	if (frontCounter > 0) {
		FrontMean = FrontMean / frontCounter;
	}
	else {
		FrontMean = 0;
	}
	if (backCounter > 0) {
		backMean = backMean / backCounter;
	}
	else {
		backMean = 0;
	}

	for (int i = 0; i < neiborPixels.rows; i++) {
		for (int j = 0; j < neiborPixels.cols; j++)
		{
			Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
			if (Edge.at<uchar>(y + i, x + j) == 1)
				frontVariance += (FrontMean - pixelColor).dot(FrontMean - pixelColor);
			else
				backVariance += (pixelColor - backMean).dot(pixelColor - backMean);
		}
	}

	if (frontCounter >0) {
		frontVariance = frontVariance / frontCounter;
	}
	else {
		frontVariance = 0;
	}
	if (backCounter > 0) {
		backVariance = backVariance / backCounter;
	}
	else {
		backVariance = 0;
	}
	res.backMean = backMean;
	res.backVar = backVariance;
	res.frontMean = FrontMean;
	res.frontVar = frontVariance;
}


double BorderMatting::Gaussian(double x, double mean, double sigma) {
    if (sigma == 0) {
        return 0;
    }
	double res = 1.0 / (pow(sigma, 0.5)*pow(2.0*PI, 0.5))* exp(-(pow(x - mean, 2.0) / (2.0*sigma)));
	return res;
}


//�����й�ʽ15��1��
double BorderMatting::Mmean(double alfa, double Fmean, double Bmean) {
	return (1.0 - alfa)*Bmean + alfa*Fmean;
}


//�����й�ʽ15��2��
double BorderMatting::Mvar(double alfa, double Fvar, double Bvar) {
	return (1.0 - alfa)*(1.0 - alfa)*Bvar + alfa*alfa*Fvar;
}



//sigmoid����,����soft step-function������ͼ6.c)
double BorderMatting::Sigmoid(double dis, double deltaCenter, double sigma) {
	// if (dis < deltaCenter - sigma / 2)
	// 	return 0;
	// if (dis >= deltaCenter + sigma / 2)
	// 	return 1;
	double res = -(dis - deltaCenter) / sigma;
	res = exp(res);
	res = 1.0 / (1.0 + res);
	return res;
}


double BorderMatting::dataTerm(point p, uchar z, int delta, int sigma, Info &para) {
	double alpha = Sigmoid(p.dis,delta,sigma);
	double MmeanValue = Mmean(alpha, valueColor2Gray(para.frontMean), valueColor2Gray(para.backMean));
	double MvarValue  = Mvar(alpha, para.frontVar, para.backVar);
	double D = Gaussian(z, MmeanValue, MvarValue);
	D = -log(D) / log(2.0);
	return D;
}



uchar BorderMatting::valueColor2Gray(Vec3b color)
{ 
	// Y <- 0.299R + 0.587G + 0.114B
	return (color[2] * 299 + color[1] * 587 + color[0] * 114 ) / 1000; 
}


// ���������е�������delta��level��30��sigma��level��10��������D������D��Сʱ��delta��sigma
void BorderMatting::Run()
{
	
	int delta = MAXDELTA / 2; 
	int sigma = MAXSIGMA / 2;

	for (int i = 0; i < contourVector.size(); i++)
	{
		Info info;
		computeMeanVariance(contourVector[i].pointInfo, info);
		contourVector[i].pointInfo.nearbyInfo = info;
		for (int j = 0; j < contourVector[i].neighbor.size(); j++)
		{
			point &p = contourVector[i].neighbor[j];
			computeMeanVariance(p, info);
			p.nearbyInfo = info;
		}

		// ����������С��
		double min = INFINITY;
		for (int deltalevel = 0; deltalevel < 30; deltalevel++) {
			for (int sigmalevel = 0; sigmalevel < 10; sigmalevel++)
			{
				double grayValue = valueColor2Gray(Image.at<Vec3b>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x));
				double D = dataTerm(contourVector[i].pointInfo, grayValue, deltalevel, sigmalevel, contourVector[i].pointInfo.nearbyInfo);
				for (int j = 0; j < contourVector[i].neighbor.size(); j++)
				{
					point &p = contourVector[i].neighbor[j];
					D += dataTerm(p, valueColor2Gray(Image.at<Vec3b>(p.y, p.x)), deltalevel, sigmalevel, p.nearbyInfo);
				}
				double V = lamda1 * (deltalevel - delta)*(deltalevel - delta) + lamda2 * (sigma - sigmalevel)*(sigma - sigmalevel); // �������Ĺ�ʽ 13
				if (D + V < min)
				{
					min = D + V;
					contourVector[i].pointInfo.delta = deltalevel;
					contourVector[i].pointInfo.sigma = sigmalevel;
				}
			}
		}
		sigma = contourVector[i].pointInfo.sigma;
		delta = contourVector[i].pointInfo.delta;
		contourVector[i].pointInfo.alpha = Sigmoid(0, delta, sigma);
		for (int j = 0; j < contourVector[i].neighbor.size(); j++)
		{
			point &p = contourVector[i].neighbor[j];
			p.alpha = Sigmoid(p.dis, delta, sigma);
            cout << p.dis << ' ' << delta << ' ' << sigma << ' ' << p.alpha << endl;
		}
	}

	Mat alphaMask = Mat(Mask.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < Mask.rows; i++) {
		for (int j = 0; j < Mask.cols; j++) {
			alphaMask.at<uchar>(i, j) = Mask.at<uchar>(i, j) * 255;
		}
	}

	for (int i = 0; i < contourVector.size(); i++) {
		alphaMask.at<uchar>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x) = (uchar)(contourVector[i].pointInfo.alpha * 255);
        float max_alpha = 0;
		for (int j = 0; j < contourVector[i].neighbor.size(); j++) {
			point &p = contourVector[i].neighbor[j];
            if (p.alpha > max_alpha) {
                max_alpha = p.alpha;
            }
        }
		for (int j = 0; j < contourVector[i].neighbor.size(); j++) {
			point &p = contourVector[i].neighbor[j];
			alphaMask.at<uchar>(p.y, p.x) = (uchar)(p.alpha / max_alpha * 255);
			// alphaMask.at<uchar>(p.y, p.x) = (uchar)(p.alpha * 255);
		}
	}
    imwrite("/mnt/lustre/wangluting/GrabCut-BorderMatting/test.png", alphaMask);

 	// Mat alphaMask = Mat(Mask.size(), CV_32FC1, Scalar(0));
 	// for (int i = 0; i < Mask.rows; i++) 
 	// {
 	// 	for (int j = 0; j < Mask.cols; j++) 
 	// 	{
 	// 		alphaMask.at<float>(i, j) = Mask.at<uchar>(i, j);
 	// 	}
 	// }
 
 	// for (int i = 0; i < contourVector.size(); i++)
 	// {
 	// 	alphaMask.at<float>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x) = contourVector[i].pointInfo.alpha;
 	// 	for (int j = 0; j < contourVector[i].neighbor.size(); j++)
 	// 	{
 	// 		point &p = contourVector[i].neighbor[j];
 	// 		alphaMask.at<float>(p.y, p.x) = p.alpha;
 	// 	}
 	// }
 
 	Mat rst = Mat(Image.size(), CV_8UC3);
 	for (int i = 0; i < rst.rows; i++) {
 		for (int j = 0; j < rst.cols; j++)
 		{
 			if (alphaMask.at<uchar>(i, j)) {
                cout << (float) alphaMask.at<uchar>(i, j) / 255.0 << endl;
 				rst.at<Vec3b>(i, j) = Vec3b(Image.at<Vec3b>(i, j)[0] * alphaMask.at<uchar>(i, j) / 255, Image.at<Vec3b>(i, j)[1] * alphaMask.at<uchar>(i, j) / 255, Image.at<Vec3b>(i, j)[2] * alphaMask.at<uchar>(i, j) / 255);
 			} else {
 				rst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
 			}
 		}
 	}
 	// imshow("bordingmatting", rst);
    imwrite("/mnt/lustre/wangluting/GrabCut-BorderMatting/test.jpg", rst);

	std::cout << "bm done!" << std::endl;
}

void BorderMatting::showEdge() {
	if (!haveEdge) {
		return;
	}
	imshow("canny",Edge);
}