// Copyright 2008 Isis Innovation Limited
#include "System.h"
#include "OpenGL.h"
#include <gvars3/instances.h>
#include <stdlib.h>
#include "ATANCamera.h"
#include "MapMaker.h"
#include "Tracker.h"
#include "ARDriver.h"
#include "MapViewer.h"

#include <cvd/image_io.h>
#include <fstream>

using namespace CVD;
using namespace std;
using namespace GVars3;


System::System()
  : mGLWindow(mVideoSource.Size(), "PTAM"), currPicIndex(0), isTempJudge(true)
{
  GUI.RegisterCommand("exit", GUICommandCallBack, this);
  GUI.RegisterCommand("quit", GUICommandCallBack, this);
  
  mimFrameBW.resize(mVideoSource.Size());
  mimFrameRGB.resize(mVideoSource.Size());
  // First, check if the camera is calibrated.
  // If not, we need to run the calibration widget.
  Vector<NUMTRACKERCAMPARAMETERS> vTest;
  
  vTest = GV3::get<Vector<NUMTRACKERCAMPARAMETERS> >("Camera.Parameters", ATANCamera::mvDefaultParams, HIDDEN);
  mpCamera = new ATANCamera("Camera");
  Vector<2> v2;
  if(v2==v2) ;
  if(vTest == ATANCamera::mvDefaultParams)
    {
      cout << endl;
      cout << "! Camera.Parameters is not set, need to run the CameraCalibrator tool" << endl;
      cout << "  and/or put the Camera.Parameters= line into the appropriate .cfg file." << endl;
      exit(1);
    }
  
  mpMap = new Map;
  mpMapMaker = new MapMaker(*mpMap, *mpCamera);
  mpTracker = new Tracker(mVideoSource.Size(), *mpCamera, *mpMap, *mpMapMaker);
  mpARDriver = new ARDriver(*mpCamera, mVideoSource.Size(), mGLWindow);
  mpMapViewer = new MapViewer(*mpMap, mGLWindow);
  
  GUI.ParseLine("GLWindow.AddMenu Menu Menu");
  GUI.ParseLine("Menu.ShowMenu Root");
  GUI.ParseLine("Menu.AddMenuButton Root Reset Reset Root");
  GUI.ParseLine("Menu.AddMenuButton Root Spacebar PokeTracker Root");
  GUI.ParseLine("DrawAR=0");
  GUI.ParseLine("DrawMap=0");
  GUI.ParseLine("Menu.AddMenuToggle Root \"View Map\" DrawMap Root");
  GUI.ParseLine("Menu.AddMenuToggle Root \"Draw AR\" DrawAR Root");
  
  mbDone = false;
};


void System::Run()
{
    while(!mbDone)
    {
        //cout << " System " << endl;
        // We use two versions of each video frame:
        // One black and white (for processing by the tracker etc)
        // and one RGB, for drawing.

        // Grab new video frame...
        mVideoSource.GetAndFillFrameBWandRGB(mimFrameBW, mimFrameRGB);

        static bool bFirstFrame = true;
        if(bFirstFrame)
        {
            mpARDriver->Init();
            bFirstFrame = false;

            LoadSVOGreenImages ("/home/zyx/project4slam/rpg_svo/svo/Datasets/sin2_tex2_h1_v8_d/img", vstrImageFilenames);
//            LoadSVOImages ("/home/zyx/Downloads/airground_rig_s3_2013-03-18_21-38-48_SVO", vstrImageFilenames);
//            LoadSVOImages ("/home/zyx/Downloads/mav_circle", vstrImageFilenames);
//            LoadImages ("/home/zyx/Downloads/MH_01_easy/mav0/cam0/data",
//                        "/home/zyx/catkin_ws_orb/src/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH01.txt",
//                        vstrImageFilenames);

//            LoadImages ("/home/zyx/Downloads/MH_03_medium/mav0/cam0/data",
//                        "/home/zyx/catkin_ws_orb/src/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH02.txt",
//                        vstrImageFilenames);
//            LoadImages ("/home/zyx/Downloads/MH_02_easy/mav0/cam0/data",
//                        "/home/zyx/catkin_ws_orb/src/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH02.txt",
//                        vstrImageFilenames);

//            LoadImages ("/home/zyx/Downloads/MH_05_diffcult/mav0/cam0/data",
//                        "/home/zyx/catkin_ws_orb/src/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH05.txt",
//                        vstrImageFilenames);

//            LoadImages ("/home/zyx/Downloads/V1_01_easy/mav0/cam0/data",
//                        "/home/zyx/catkin_ws_orb/src/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/V101.txt",
//                        vstrImageFilenames);
//            cout << "获取图片数: " << vstrImageFilenames.size() << endl;

//            LoadDataImages ("/home/zyx/Downloads/Data", vstrImageFilenames);
//            LoadData4PTAMImages ("/home/zyx/Downloads/Data4PTAM", vstrImageFilenames);

        }

        Image<byte> img;//灰度图形式读取
        //CVD::img_load(img,"/home/zyx/Downloads/MH_01_easy/mav0/cam0/data/1403636579763555584.png");


        //参考:http://www.edwardrosten.com/cvd/cvd/html/namespaceCVD.html
        CVD::img_load(img, vstrImageFilenames[currPicIndex++]);
        if (currPicIndex == 2) {
            CVD::img_save(img, "/home/zyx/img.png");//保存一张图片看看
        }
        mimFrameBW = img;




        //cout << "图片大小为: " << img.size() << endl;

//        if (isTempJudge) {

//            cout << "图片大小为: " << img.size() << endl;
//            cv::Mat image(480, 752, CV_8UC1, img.data());
//            cout << "image.size: " << image.rows << " " << image.cols << endl;
//            imwrite("111111.png", image);
//            cv::imshow("war", image);
//            isTempJudge = false;

//            cv::Mat src = cv::imread("/home/zyx/Downloads/MH_01_easy/mav0/cam0/data/1403636579763555584.png");
//            cv::imshow("2222", src);
//        }


        mGLWindow.SetupViewport();
        mGLWindow.SetupVideoOrtho();
        mGLWindow.SetupVideoRasterPosAndZoom();

        if(!mpMap->IsGood())
            mpARDriver->Reset();

        static gvar3<int> gvnDrawMap("DrawMap", 0, HIDDEN|SILENT);
        static gvar3<int> gvnDrawAR("DrawAR", 0, HIDDEN|SILENT);

        bool bDrawMap = mpMap->IsGood() && *gvnDrawMap;
        bool bDrawAR = mpMap->IsGood() && *gvnDrawAR;

        mpTracker->TrackFrame(mimFrameBW, !bDrawAR && !bDrawMap);

        if(bDrawMap)
            mpMapViewer->DrawMap(mpTracker->GetCurrentPose());
        else if(bDrawAR)
            mpARDriver->Render(mimFrameRGB, mpTracker->GetCurrentPose());

        //      mGLWindow.GetMousePoseUpdate();
        string sCaption;
        if(bDrawMap)
            sCaption = mpMapViewer->GetMessageForUser();
        else
            sCaption = mpTracker->GetMessageForUser();
        mGLWindow.DrawCaption(sCaption);
        mGLWindow.DrawMenus();
        mGLWindow.swap_buffers();
        mGLWindow.HandlePendingEvents();
    }
}

void System::GUICommandCallBack(void *ptr, string sCommand, string sParams)
{
  if(sCommand=="quit" || sCommand == "exit")
    static_cast<System*>(ptr)->mbDone = true;
}


//modified by zyx 2017.8.13
void System::LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vstrImages.reserve(5000);//使用reserve来避免不必要的重新分配
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");//存放图片路径
        }
    }
}

void System::LoadSVOGreenImages(const string &strImagePath, vector<string> &vstrImages)
{
    vstrImages.reserve(200);//使用reserve来避免不必要的重新分配

    for(int ii = 2; ii <= 187; ii++)
    {
        stringstream ss;
        ss << ii;
        string str = "";
        if (ii < 10) {
            str = "frame_00000" + ss.str();
        } else if (ii < 100) {
            str = "frame_0000" + ss.str();
        } else if (ii < 1000) {
            str = "frame_000" + ss.str();
        }
        vstrImages.push_back(strImagePath + "/" + str + "_0.png");//存放图片路径
    }

    cout << vstrImages[0] << endl;
    cout << vstrImages[11] << endl;
    cout << vstrImages[111] << endl;
}

void System::LoadSVOImages(const string &strImagePath, vector<string> &vstrImages)
{
    vstrImages.reserve(1400);//使用reserve来避免不必要的重新分配

    for(int ii = 0; ii <= 1360; ii++)
    {
        stringstream ss;
        ss << ii;
        string str = "";
        if (ii < 10) {
            str = "frame000" + ss.str();
        } else if (ii < 100) {
            str = "frame00" + ss.str();
        } else if (ii < 1000) {
            str = "frame0" + ss.str();
        } else {
            str = "frame" + ss.str();
        }
        vstrImages.push_back(strImagePath + "/" + str + ".jpg");//存放图片路径
    }

    cout << vstrImages[0] << endl;
    cout << vstrImages[11] << endl;
    cout << vstrImages[111] << endl;
    cout << vstrImages[1111] << endl;
}

void System::LoadDataImages(const string &strImagePath, vector<string> &vstrImages)
{
    vstrImages.reserve(810);//使用reserve来避免不必要的重新分配

    for(int ii = 1; ii <= 800; ii++)
    {
        stringstream ss;
        ss << ii;
        string str = ss.str();
        vstrImages.push_back(strImagePath + "/" + str + ".png");//存放图片路径
    }

    cout << vstrImages[0] << endl;
    cout << vstrImages[11] << endl;
    cout << vstrImages[111] << endl;
}


void System::LoadData4PTAMImages(const string &strImagePath, vector<string> &vstrImages)
{
    vstrImages.reserve(625);//使用reserve来避免不必要的重新分配

    for(int ii = 0; ii <= 623; ii++)
    {
        stringstream ss;
        ss << ii;
        string str = ss.str();
        vstrImages.push_back(strImagePath + "/" + str + ".png");//存放图片路径
    }

    cout << vstrImages[0] << endl;
    cout << vstrImages[11] << endl;
    cout << vstrImages[111] << endl;
}





