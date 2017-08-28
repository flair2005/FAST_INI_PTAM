// -*- c++ -*-
// Copyright 2008 Isis Innovation Limited
//
// System.h
//
// Defines the System class
//
// This stores the main functional classes of the system, like the
// mapmaker, map, tracker etc, and spawns the working threads.
//
#ifndef __SYSTEM_H
#define __SYSTEM_H
#include "VideoSource.h"
#include "GLWindow2.h"

#include <cvd/image.h>
#include <cvd/rgb.h>
#include <cvd/byte.h>

class ATANCamera;
class Map;
class MapMaker;
class Tracker;
class ARDriver;
class MapViewer;

class System
{
public:
  System();
  void Run();
  
private:
  VideoSource mVideoSource;
  GLWindow2 mGLWindow;
  CVD::Image<CVD::Rgb<CVD::byte> > mimFrameRGB;
  CVD::Image<CVD::byte> mimFrameBW;
  
  Map *mpMap; 
  MapMaker *mpMapMaker; 
  Tracker *mpTracker; 
  ATANCamera *mpCamera;
  ARDriver *mpARDriver;
  MapViewer *mpMapViewer;
  
  bool mbDone;

  static void GUICommandCallBack(void* ptr, std::string sCommand, std::string sParams);

  //added by zyx
  void LoadImages(const std::string &strImagePath, const std::string &strPathTimes,
                  std::vector<std::string> &vstrImages);
  void LoadSVOImages(const std::string &strImagePath, std::vector<std::string> &vstrImages);
  void LoadDataImages(const std::string &strImagePath, std::vector<std::string> &vstrImages);
  void LoadData4PTAMImages(const std::string &strImagePath, std::vector<std::string> &vstrImages);
  int currPicIndex;//当前图片索引
  std::vector<std::string> vstrImageFilenames;//存储图片名字
  bool isTempJudge;//用来测试CVD::image与cv::Mat之间的转换
};



#endif
