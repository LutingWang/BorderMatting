#include <iostream>
#include <opencv2/opencv.hpp>
#include "BorderMatting.h"

int main(int argc, char* argv[])
{
	string img_name = "/mnt/lustre/wangluting/GrabCut-BorderMatting/media/apple.jpg";
	//string filename = "./timg2.jpg";
	//string filename = "./tree.png";
	Mat img= imread(img_name, 1 );
	if(img.empty() )
	{
		cout << "couldn't read image " << img_name << endl;
		return 1;
	}
    Mat mask, bgdModel, fgdModel;
    mask.create(img.size(), CV_8UC1);
    grabCut(img, mask, Rect(20, 100, 240, 200), bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
    imwrite("testmask.jpg", mask);
    // string mask_name = "/mnt/lustre/wangluting/GrabCut-BorderMatting/media/mask.png";
    // Mat mask = imread(mask_name, 1);
    // cvtColor(mask, mask, COLOR_BGR2GRAY);
    // if(mask.empty() )
    // {
    //     cout << "couldn't read mask " << mask_name << endl;
    //     return 1;
    // }
    BorderMatting bm;
    bm.Initialize(img, mask);
    bm.Run();
    return 0;
}