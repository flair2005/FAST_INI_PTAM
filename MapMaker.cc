// Copyright 2008 Isis Innovation Limited
#include "MapMaker.h"
#include "MapPoint.h"
#include "Bundle.h"
#include "PatchFinder.h"
#include "SmallMatrixOpts.h"
#include "HomographyInit.h"

#include <cvd/vector_image_ref.h>
#include <cvd/vision.h>
#include <cvd/image_interpolate.h>

#include <TooN/SVD.h>
#include <TooN/SymEigen.h>

#include <gvars3/instances.h>
#include <fstream>
#include <algorithm>

#include <opencv2/core/core_c.h>
#include "lsd.h"
#include "MSAC.h"
#include <random>


#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace CVD;
using namespace std;
using namespace GVars3;

// Constructor sets up internal reference variable to Map.
// Most of the intialisation is done by Reset()..
MapMaker::MapMaker(Map& m, const ATANCamera &cam)
  : mMap(m), mCamera(cam)
{
  mbResetRequested = false;
  Reset();
  start(); // This CVD::thread func starts the map-maker thread with function run()
  GUI.RegisterCommand("SaveMap", GUICommandCallBack, this);
  GV3::Register(mgvdWiggleScale, "MapMaker.WiggleScale", 0.1, SILENT); // Default to 10cm between keyframes
};

void MapMaker::Reset()
{
  // This is only called from within the mapmaker thread...
  mMap.Reset();
  mvFailureQueue.clear();
  while(!mqNewQueue.empty()) mqNewQueue.pop();
  mMap.vpKeyFrames.clear(); // TODO: actually erase old keyframes
  mvpKeyFrameQueue.clear(); // TODO: actually erase old keyframes
  mbBundleRunning = false;
  mbBundleConverged_Full = true;
  mbBundleConverged_Recent = true;
  mbResetDone = true;
  mbResetRequested = false;
  mbBundleAbortRequested = false;
}

// CHECK_RESET is a handy macro which makes the mapmaker thread stop
// what it's doing and reset, if required.
#define CHECK_RESET if(mbResetRequested) {Reset(); continue;};

void MapMaker::run()
{

#ifdef WIN32
  // For some reason, I get tracker thread starvation on Win32 when
  // adding key-frames. Perhaps this will help:
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#endif

  while(!shouldStop())  // ShouldStop is a CVD::Thread func which return true if the thread is told to exit.
    {
      CHECK_RESET;
      sleep(5); // Sleep not really necessary, especially if mapmaker is busy
      CHECK_RESET;
      
      // Handle any GUI commands encountered..
      while(!mvQueuedCommands.empty())
	{
	  GUICommandHandler(mvQueuedCommands.begin()->sCommand, mvQueuedCommands.begin()->sParams);
	  mvQueuedCommands.erase(mvQueuedCommands.begin());
	}
      
      if(!mMap.IsGood())  // Nothing to do if there is no map yet!
	continue;
      
      // From here on, mapmaker does various map-maintenance jobs in a certain priority
      // Hierarchy. For example, if there's a new key-frame to be added (QueueSize() is >0)
      // then that takes high priority.
      
      CHECK_RESET;
      // Should we run local bundle adjustment?
      if(!mbBundleConverged_Recent && QueueSize() == 0)  
	BundleAdjustRecent();   
      
      CHECK_RESET;
      // Are there any newly-made map points which need more measurements from older key-frames?
      if(mbBundleConverged_Recent && QueueSize() == 0)
	ReFindNewlyMade();  
      
      CHECK_RESET;
      // Run global bundle adjustment?
      if(mbBundleConverged_Recent && !mbBundleConverged_Full && QueueSize() == 0)
	BundleAdjustAll();
      
      CHECK_RESET;
      // Very low priorty: re-find measurements marked as outliers
      if(mbBundleConverged_Recent && mbBundleConverged_Full && rand()%20 == 0 && QueueSize() == 0)
	ReFindFromFailureQueue();
      
      CHECK_RESET;
      HandleBadPoints();
      
      CHECK_RESET;
      // Any new key-frames to be added?
      if(QueueSize() > 0)
	AddKeyFrameFromTopOfQueue(); // Integrate into map data struct, and process
    }
}


// Tracker calls this to demand a reset
void MapMaker::RequestReset()
{
  mbResetDone = false;
  mbResetRequested = true;
}

bool MapMaker::ResetDone()
{
  return mbResetDone;
}

// HandleBadPoints() Does some heuristic checks on all points in the map to see if 
// they should be flagged as bad, based on tracker feedback.
void MapMaker::HandleBadPoints()
{
  // Did the tracker see this point as an outlier more often than as an inlier?
  for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
    {
      MapPoint &p = *mMap.vpPoints[i];
      if(p.nMEstimatorOutlierCount > 20 && p.nMEstimatorOutlierCount > p.nMEstimatorInlierCount)
	p.bBad = true;
    }
  
  // All points marked as bad will be erased - erase all records of them
  // from keyframes in which they might have been measured.
  for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
    if(mMap.vpPoints[i]->bBad)
      {
	MapPoint *p = mMap.vpPoints[i];
	for(unsigned int j=0; j<mMap.vpKeyFrames.size(); j++)
	  {
	    KeyFrame &k = *mMap.vpKeyFrames[j];
	    if(k.mMeasurements.count(p))
	      k.mMeasurements.erase(p);
	  }
      }
  // Move bad points to the trash list.
  mMap.MoveBadPointsToTrash();
}

MapMaker::~MapMaker()
{
  mbBundleAbortRequested = true;
  stop(); // makes shouldStop() return true
  cout << "Waiting for mapmaker to die.." << endl;
  join();
  cout << " .. mapmaker has died." << endl;
}


// Finds 3d coords of point in reference frame B from two z=1 plane projections
//三角测量恢复出3D坐标点
Vector<3> MapMaker::ReprojectPoint(SE3<> se3AfromB, const Vector<2> &v2A, const Vector<2> &v2B)
{
  Matrix<3,4> PDash;
  PDash.slice<0,0,3,3>() = se3AfromB.get_rotation().get_matrix();
  PDash.slice<0,3,3,1>() = se3AfromB.get_translation().as_col();
  
  Matrix<4> A;
  //cout << "矩阵: " << A << endl;
  A[0][0] = -1.0; A[0][1] =  0.0; A[0][2] = v2B[0]; A[0][3] = 0.0;
  A[1][0] =  0.0; A[1][1] = -1.0; A[1][2] = v2B[1]; A[1][3] = 0.0;
  A[2] = v2A[0] * PDash[2] - PDash[0];
  A[3] = v2A[1] * PDash[2] - PDash[1];

  SVD<4,4> svd(A);
  Vector<4> v4Smallest = svd.get_VT()[3];
  if(v4Smallest[3] == 0.0)
    v4Smallest[3] = 0.00001;
  return project(v4Smallest);
}

//static bool MapMaker::less_int(MapPoint* a,MapPoint* b){

//        return a->v3WorldPos[0] < b->v3WorldPos[1];

//}

// InitFromStereo() generates the initial match from two keyframes
// and a vector of image correspondences. Uses the 
bool MapMaker::InitFromStereo(KeyFrame &kF,
                              KeyFrame &kS,
                              vector<pair<ImageRef, ImageRef> > &vTrailMatches,
                              SE3<> &se3TrackerPose)
{
//    if (!mMap.vpPoints.empty()) {
//        cout << "mMap.vpPoints.size() " << mMap.vpPoints.size() << endl;
//    }

    //cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

    cout << "  MapMakerqqq: made initial map with " << mMap.vpPoints.size() << " points." << endl;
    mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

    mCamera.SetImageSize(kF.aLevels[0].im.size());

    cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

    vector<HomographyMatch> vMatches;
    for(unsigned int i=0; i<vTrailMatches.size(); i++)
    {
        HomographyMatch m;
        m.v2CamPlaneFirst = mCamera.UnProject(vTrailMatches[i].first);
//        cout << "m.v2CamPlaneFirst0: " << endl << m.v2CamPlaneFirst << endl;
        m.v2CamPlaneSecond = mCamera.UnProject(vTrailMatches[i].second);
//        cout << "m.v2CamPlaneSecond0: " << endl << m.v2CamPlaneSecond << endl;
        m.m2PixelProjectionJac = mCamera.GetProjectionDerivs();
//        cout << "m.m2PixelProjectionJac0: " << endl << m.m2PixelProjectionJac << endl;
        vMatches.push_back(m);
    }

    cout << "vMatches.size()0 " << vMatches.size() << endl;


    SE3<> se3;
    bool bGood;
    HomographyInit HomographyInit;
    bGood = HomographyInit.Compute2(vMatches, 5.0, se3, 0);
    cout << "se3: " << se3 << endl;
    cout << "vMatches.size()1 " << vMatches.size() << endl;
    if(!bGood)
    {
        cout << "  Could not init from stereo pair, try again." << endl;
        return false;
    }

    // Check that the initialiser estimated a non-zero baseline
    double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
    if(dTransMagn == 0)
    {
        cout << "  Estimated zero baseline from stereo pair, try again." << endl;
        return false;
    }
    // change the scale of the map so the second camera is wiggleScale away from the first
    se3.get_translation() *= mdWiggleScale/dTransMagn;


    KeyFrame *pkFirst = new KeyFrame();
    KeyFrame *pkSecond = new KeyFrame();
    *pkFirst = kF;
    *pkSecond = kS;

    pkFirst->bFixed = true;
    pkFirst->se3CfromW = SE3<>();

    pkSecond->bFixed = false;
    pkSecond->se3CfromW = se3;

    // Construct map from the stereo matches.
    PatchFinder finder;

    cout << "vMatches.size()3 " << vMatches.size() << endl;
    cout << "se3: " << se3 << endl;

    bool first = true;
    for(unsigned int i=0; i<vMatches.size(); i++)
    {
        MapPoint *p = new MapPoint();

        // Patch source stuff:
        p->pPatchSourceKF = pkFirst;
        p->nSourceLevel = 0;
        p->v3Normal_NC = makeVector( 0,0,-1);
        p->irCenter = vTrailMatches[i].first;
        p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
//        if (i == 0) {
//            cout << mCamera.UnProject(p->irCenter)[0] << " " << mCamera.UnProject(p->irCenter)[1] << " "
//                                                      << p->v3Center_NC[0] << " " << p->v3Center_NC[1] << " " << p->v3Center_NC[2] << endl;
//        }
        p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
        p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
        normalize(p->v3Center_NC);
        normalize(p->v3OneDownFromCenter_NC);
        normalize(p->v3OneRightFromCenter_NC);
        p->RefreshPixelVectors();

        // Do sub-pixel alignment on the second image
        finder.MakeTemplateCoarseNoWarp(*p);
        finder.MakeSubPixTemplate();
        finder.SetSubPixPos(vec(vTrailMatches[i].second));
        bool bGood = finder.IterateSubPixToConvergence(*pkSecond,10);
        if(!bGood)
        {
            delete p; continue;
        }

        // Triangulate point:
        Vector<2> v2SecondPos = finder.GetSubPixPos();
//        cout << vMatches[i].v2CamPlaneFirst[0] << " " << vMatches[i].v2CamPlaneFirst[1] << endl;第一帧
//        cout << v2SecondPos[0] << " " << v2SecondPos[1] << endl;
//        cout << mCamera.UnProject(v2SecondPos)[0] << " " << mCamera.UnProject(v2SecondPos)[1] << endl;
//        if (first) {
//            cout << "v2SecondPos: " << v2SecondPos << endl;
//        }
        p->v3WorldPos = ReprojectPoint(se3, mCamera.UnProject(v2SecondPos), vMatches[i].v2CamPlaneFirst);
//        cout << "vMatches[i].v2CamPlaneFirst: " << endl << vMatches[i].v2CamPlaneFirst << endl;
        if(p->v3WorldPos[2] < 0.0)
        {
            delete p; continue;
        }
        //cout << p->v3WorldPos[0] << " " << p->v3WorldPos[1] << " " << p->v3WorldPos[2] << endl;
//        for (int i = 1; i < 432; i++) {
//            p->v3WorldPos[0] = mCamera.UnProject(v2SecondPos)[0];
//            p->v3WorldPos[1] = mCamera.UnProject(v2SecondPos)[1];
//            p->v3WorldPos[1] = 0.5;
//        }

        // Not behind map? Good, then add to map.
        p->pMMData = new MapMakerData();
        mMap.vpPoints.push_back(p);

        // Construct first two measurements and insert into relevant DBs:
        Measurement mFirst;
        mFirst.nLevel = 0;
        mFirst.Source = Measurement::SRC_ROOT;
        mFirst.v2RootPos = vec(vTrailMatches[i].first);
        mFirst.bSubPix = true;
        pkFirst->mMeasurements[p] = mFirst;
        p->pMMData->sMeasurementKFs.insert(pkFirst);

        Measurement mSecond;
        mSecond.nLevel = 0;
        mSecond.Source = Measurement::SRC_TRAIL;
        mSecond.v2RootPos = finder.GetSubPixPos();
        mSecond.bSubPix = true;
        pkSecond->mMeasurements[p] = mSecond;
        p->pMMData->sMeasurementKFs.insert(pkSecond);
    }


    mMap.vpKeyFrames.push_back(pkFirst);
    mMap.vpKeyFrames.push_back(pkSecond);
    pkFirst->MakeKeyFrame_Rest();
    pkSecond->MakeKeyFrame_Rest();

    cout << "  MapMaker110: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    for(int i=0; i<5; i++)
        BundleAdjustAll();

    cout << "  MapMaker119: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    // Estimate the feature depth distribution in the first two key-frames
    // (Needed for epipolar search)
    RefreshSceneDepth(pkFirst);
    RefreshSceneDepth(pkSecond);
    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;


//    std::vector<CVD::ImageRef> vMapPointsCandidates;//存储Candidates的临时变量
//    AddSomeMapPointsForFast (vMapPointsCandidates, 0);
//    cout << "vMapPointsCandidates0: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 3);
//    cout << "vMapPointsCandidates1: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 1);
//    cout << "vMapPointsCandidates2: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 2);
//    cout << "vMapPointsCandidates3: " << vMapPointsCandidates.size() << endl;

    //cout << "  MapMaker0: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    AddSomeMapPoints(0);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[0].vCandidates.size() << endl;
    AddSomeMapPoints(3);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[3].vCandidates.size() << endl;
    AddSomeMapPoints(1);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[1].vCandidates.size() << endl;
    AddSomeMapPoints(2);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[2].vCandidates.size() << endl;
    //cout << "SUM: " << pkSecond->aLevels[0].vCandidates.size() + pkSecond->aLevels[1].vCandidates.size()
    //        + pkSecond->aLevels[2].vCandidates.size() + pkSecond->aLevels[3].vCandidates.size() << endl;



    mbBundleConverged_Full = false;
    mbBundleConverged_Recent = false;

    while(!mbBundleConverged_Full)
    {
        BundleAdjustAll();
        if(mbResetRequested)
            return false;
    }

    // Rotate and translate the map so the dominant plane is at z=0:
    ApplyGlobalTransformationToMap(CalcPlaneAligner());
    mMap.bGood = true;
    se3TrackerPose = pkSecond->se3CfromW;

    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;
//    cout << "qqqqqqq " << mMap.vpPoints[0]->v3WorldPos[0] << endl;
//    for (int i = 0; i < mMap.vpPoints.size(); i++) {
//        cout << "v3WorldPos " << i << ": " << mMap.vpPoints[i]->v3WorldPos[0] << " "
//             << mMap.vpPoints[i]->v3WorldPos[1] << " " << mMap.vpPoints[i]->v3WorldPos[2] << endl;
//    }

//    int cc = 0;
//    int cc2 = 0;
//    for (int i = 0; i < mMap.vpPoints.size(); i++) {
//        int x1 = mMap.vpPoints[i]->v3WorldPos[0];
//        for (int j = i+1; j < mMap.vpPoints.size(); j++) {
//            if (  (x1 - mMap.vpPoints[j]->v3WorldPos[0] > 0)
////                 (x1 - mMap.vpPoints[j]->v3WorldPos[0] > -0.001 && x1 - mMap.vpPoints[j]->v3WorldPos[0] < 0)
//                    /*&& mMap.vpPoints[i]->v3WorldPos[1] == mMap.vpPoints[j]->v3WorldPos[1]*/) {
//                cc ++;
//                break;
//            }
//            cc2 ++;
//        }
//    }
//    cout << "cc: " << cc << endl;
//    cout << "cc2: " << cc2 << endl;

//    sort(mMap.vpPoints.begin(), mMap.vpPoints.end(), less_int);
//    //cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;
//    int cc = 0;
//    for (int k = 0; k < mMap.vpPoints.size(); k++) {
//        MapPoint* a = mMap.vpPoints[0];
//        MapPoint* b = mMap.vpPoints[1];
//        if ( a->v3WorldPos[0] == b->v3WorldPos[0] && a->v3WorldPos[1] == b->v3WorldPos[1] ) {
//            continue;
//        } else
//            cc ++;
//    }
//    cout << "cc:   " << cc << endl;
//    for (int i = 0; i < mMap.vpPoints.size(); i++) {
//        MapPoint *point = mMap.vpPoints[i];
//        cout << "[" << point->v3WorldPos[0] << ", " << point->v3WorldPos[1] << ", " << point->v3WorldPos[2] << "]" << endl;
//    }

    return true;
}

// ThinCandidates() Thins out a key-frame's candidate list.
// Candidates are those salient corners where the mapmaker will attempt 
// to make a new map point by epipolar search. We don't want to make new points
// where there are already existing map points, this routine erases such candidates.
// Operates on a single level of a keyframe.
//挑出候选点
void MapMaker::ThinCandidates(KeyFrame &k, int nLevel)
{
    vector<Candidate> &vCSrc = k.aLevels[nLevel].vCandidates;
    vector<Candidate> vCGood;
    vector<ImageRef> irBusyLevelPos;
    // Make a list of `busy' image locations, which already have features at the same level
    // or at one level higher.
    for(meas_it it = k.mMeasurements.begin(); it!=k.mMeasurements.end(); it++)
    {
        if(!(it->second.nLevel == nLevel || it->second.nLevel == nLevel + 1))
            continue;
        irBusyLevelPos.push_back(ir_rounded(it->second.v2RootPos / LevelScale(nLevel)));
//        cout << "it->second.v2RootPos " << it->second.v2RootPos
//             << " (it->second.v2RootPos / LevelScale(nLevel)) " << (it->second.v2RootPos / LevelScale(nLevel));
    }

    // Only keep those candidates further than 10 pixels away from busy positions.
    unsigned int nMinMagSquared = 10*10;
    for(unsigned int i=0; i<vCSrc.size(); i++)
    {
        ImageRef irC = vCSrc[i].irLevelPos;
        bool bGood = true;
        for(unsigned int j=0; j<irBusyLevelPos.size(); j++)
        {
            ImageRef irB = irBusyLevelPos[j];
            if((irB - irC).mag_squared() < nMinMagSquared)
            {
                bGood = false;
                break;
            }
        }
        if(bGood)
            vCGood.push_back(vCSrc[i]);
    }
    vCSrc = vCGood;
}

// Adds map points by epipolar search to the last-added key-frame, at a single
// specified pyramid level. Does epipolar search in the target keyframe as closest by
// the ClosestKeyFrame function.
void MapMaker::AddSomeMapPoints(int nLevel)
{
  KeyFrame &kSrc = *(mMap.vpKeyFrames[mMap.vpKeyFrames.size() - 1]); // The new keyframe
  KeyFrame &kTarget = *(ClosestKeyFrame(kSrc));   
  Level &l = kSrc.aLevels[nLevel];

  ThinCandidates(kSrc, nLevel);
//  cout << " kSrc.aLevels[nLevel].vCandidates.size() " << kSrc.aLevels[nLevel].vCandidates.size() << endl;
//  cout << " l.vCandidates.size() " << l.vCandidates.size() << endl;
  for(unsigned int i = 0; i<l.vCandidates.size(); i++)
    AddPointEpipolar(kSrc, kTarget, nLevel, i);
};

// Rotates/translates the whole map and all keyframes
void MapMaker::ApplyGlobalTransformationToMap(SE3<> se3NewFromOld)
{
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    mMap.vpKeyFrames[i]->se3CfromW = mMap.vpKeyFrames[i]->se3CfromW * se3NewFromOld.inverse();
  
  SO3<> so3Rot = se3NewFromOld.get_rotation();
  for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
    {
      mMap.vpPoints[i]->v3WorldPos = 
	se3NewFromOld * mMap.vpPoints[i]->v3WorldPos;
      mMap.vpPoints[i]->RefreshPixelVectors();
    }
}

// Applies a global scale factor to the map
void MapMaker::ApplyGlobalScaleToMap(double dScale)
{
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    mMap.vpKeyFrames[i]->se3CfromW.get_translation() *= dScale;
  
  for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
    {
      (*mMap.vpPoints[i]).v3WorldPos *= dScale;
      (*mMap.vpPoints[i]).v3PixelRight_W *= dScale;
      (*mMap.vpPoints[i]).v3PixelDown_W *= dScale;
      (*mMap.vpPoints[i]).RefreshPixelVectors();
    }
}

// The tracker entry point for adding a new keyframe;
// the tracker thread doesn't want to hang about, so 
// just dumps it on the top of the mapmaker's queue to 
// be dealt with later, and return.
void MapMaker::AddKeyFrame(KeyFrame &k)
{
  KeyFrame *pK = new KeyFrame;
  *pK = k;
  pK->pSBI = NULL; // Mapmaker uses a different SBI than the tracker, so will re-gen its own
  mvpKeyFrameQueue.push_back(pK);
  if(mbBundleRunning)   // Tell the mapmaker to stop doing low-priority stuff and concentrate on this KF first.
    mbBundleAbortRequested = true;
}

// Mapmaker's code to handle incoming key-frames.
void MapMaker::AddKeyFrameFromTopOfQueue()
{
  if(mvpKeyFrameQueue.size() == 0)
    return;
  
  KeyFrame *pK = mvpKeyFrameQueue[0];
  mvpKeyFrameQueue.erase(mvpKeyFrameQueue.begin());
  pK->MakeKeyFrame_Rest();
  mMap.vpKeyFrames.push_back(pK);
  // Any measurements? Update the relevant point's measurement counter status map
  for(meas_it it = pK->mMeasurements.begin();
      it!=pK->mMeasurements.end();
      it++)
    {
      it->first->pMMData->sMeasurementKFs.insert(pK);
      it->second.Source = Measurement::SRC_TRACKER;
    }
  
  // And maybe we missed some - this now adds to the map itself, too.
  ReFindInSingleKeyFrame(*pK);
  
  AddSomeMapPoints(3);       // .. and add more map points by epipolar search.
  AddSomeMapPoints(0);
  AddSomeMapPoints(1);
  AddSomeMapPoints(2);
  
  mbBundleConverged_Full = false;
  mbBundleConverged_Recent = false;
}

// Tries to make a new map point out of a single candidate point
// by searching for that point in another keyframe, and triangulating
// if a match is found.
bool MapMaker::AddPointEpipolar(KeyFrame &kSrc, 
                                KeyFrame &kTarget,
                                int nLevel,
                                int nCandidate)
{
    //Image的访问方式 Image[ImageRef ir]
    //这个数据结构Image<Vector<2> > 代表了像素坐标与相机坐标的对应关系，像素坐标就是对Image的访问方式， Vector<2>存储了相机坐标系下的坐标
    static Image<Vector<2> > imUnProj;//这个应该相机归一化坐标系下的图像，也就是所有的像素点都转化到相机坐标系下了
    static bool bMadeCache = false;
    if(!bMadeCache)
    {
        imUnProj.resize(kSrc.aLevels[0].im.size());
        ImageRef ir;//像素坐标系原点

        cout << "ir0: " << ir << endl;
//        do imUnProj[ir] = mCamera.UnProject(ir);
//        while(ir.next(imUnProj.size()));
//        int num = 0;
        do
        {
            imUnProj[ir] = mCamera.UnProject(ir);
//            cout << "ir: " << ir << endl;
//            cout << "imUnProj[ir]: " << imUnProj[ir] << endl;
            //cout << "imUnProj[" << num << "]: " << imUnProj[num++][0] << endl;
            //cout << "imUnProj.size(): " << imUnProj.size() << endl;

        }
        while(ir.next(imUnProj.size()));//ir.next() 水平扫描下一个像素点 imUnProj.size()是图像的像素大小，例如[640,480]
        bMadeCache = true;
    }

    int nLevelScale = LevelScale(nLevel);
    Candidate &candidate = kSrc.aLevels[nLevel].vCandidates[nCandidate];
    ImageRef irLevelPos = candidate.irLevelPos;//在level层的像素坐标
    Vector<2> v2RootPos = LevelZeroPos(irLevelPos, nLevel);//将level层的像素坐标转换到第0层  也就是需要乘一个数

    Vector<3> v3Ray_SC = unproject(mCamera.UnProject(v2RootPos));//将第0层的像素坐标转换到矫正后的相机坐标，并且齐次化
    normalize(v3Ray_SC);//单位化
    //获取当前帧相机坐标系下的点在最近帧坐标系下的坐标
    Vector<3> v3LineDirn_TC = kTarget.se3CfromW.get_rotation() * (kSrc.se3CfromW.get_rotation().inverse() * v3Ray_SC);//当前帧的点投影到世界坐标系，在投影到最近帧的坐标系

    // Restrict epipolar search to a relatively narrow depth range
    // to increase reliability
    double dMean = kSrc.dSceneDepthMean;
    double dSigma = kSrc.dSceneDepthSigma;
    double dStartDepth = max(mdWiggleScale, dMean - dSigma);
    double dEndDepth = min(40 * mdWiggleScale, dMean + dSigma);

    //相机中心的平移O2->O1还是O1->O2
    Vector<3> v3CamCenter_TC = kTarget.se3CfromW * kSrc.se3CfromW.inverse().get_translation(); // The camera end
    //v3RayStart_TC与v3RayEnd_TC之差是什么
    Vector<3> v3RayStart_TC = v3CamCenter_TC + dStartDepth * v3LineDirn_TC;                               // the far-away end
    Vector<3> v3RayEnd_TC = v3CamCenter_TC + dEndDepth * v3LineDirn_TC;                               // the far-away end


    if(v3RayEnd_TC[2] <= v3RayStart_TC[2])  // it's highly unlikely that we'll manage to get anything out if we're facing backwards wrt the other camera's view-ray
        return false;
    if(v3RayEnd_TC[2] <= 0.0 )  return false;
    if(v3RayStart_TC[2] <= 0.0)
        v3RayStart_TC += v3LineDirn_TC * (0.001 - v3RayStart_TC[2] / v3LineDirn_TC[2]);

    Vector<2> v2A = project(v3RayStart_TC);//For a vector v of length i, return [v1,v2,⋯,vi−1]/vi.
    Vector<2> v2B = project(v3RayEnd_TC);
    Vector<2> v2AlongProjectedLine = v2A-v2B;

    if(v2AlongProjectedLine * v2AlongProjectedLine < 0.00000001)
    {
        cout << "v2AlongProjectedLine too small." << endl;
        return false;
    }
    normalize(v2AlongProjectedLine);
    Vector<2> v2Normal;
    v2Normal[0] = v2AlongProjectedLine[1];
    v2Normal[1] = -v2AlongProjectedLine[0];

    double dNormDist = v2A * v2Normal;
    if(fabs(dNormDist) > mCamera.LargestRadiusInImage() )
        return false;

    double dMinLen = min(v2AlongProjectedLine * v2A, v2AlongProjectedLine * v2B) - 0.05;
    double dMaxLen = max(v2AlongProjectedLine * v2A, v2AlongProjectedLine * v2B) + 0.05;
    if(dMinLen < -2.0)  dMinLen = -2.0;
    if(dMaxLen < -2.0)  dMaxLen = -2.0;
    if(dMinLen > 2.0)   dMinLen = 2.0;
    if(dMaxLen > 2.0)   dMaxLen = 2.0;

    // Find current-frame corners which might match this
    PatchFinder Finder;
    Finder.MakeTemplateCoarseNoWarp(kSrc, nLevel, irLevelPos);
    if(Finder.TemplateBad())  return false;

    vector<Vector<2> > &vv2Corners = kTarget.aLevels[nLevel].vImplaneCorners;
    vector<ImageRef> &vIR = kTarget.aLevels[nLevel].vCorners;
    if(!kTarget.aLevels[nLevel].bImplaneCornersCached)
    {
        for(unsigned int i=0; i<vIR.size(); i++)   // over all corners in target img..
            vv2Corners.push_back(imUnProj[ir(LevelZeroPos(vIR[i], nLevel))]);
        kTarget.aLevels[nLevel].bImplaneCornersCached = true;
    }

    int nBest = -1;
    int nBestZMSSD = Finder.mnMaxSSD + 1;
    double dMaxDistDiff = mCamera.OnePixelDist() * (4.0 + 1.0 * nLevelScale);
    double dMaxDistSq = dMaxDistDiff * dMaxDistDiff;

    for(unsigned int i=0; i<vv2Corners.size(); i++)   // over all corners in target img..
    {
        Vector<2> v2Im = vv2Corners[i];
        double dDistDiff = dNormDist - v2Im * v2Normal;
        if(dDistDiff * dDistDiff > dMaxDistSq)	continue; // skip if not along epi line
        if(v2Im * v2AlongProjectedLine < dMinLen)	continue; // skip if not far enough along line
        if(v2Im * v2AlongProjectedLine > dMaxLen)	continue; // or too far
        int nZMSSD = Finder.ZMSSDAtPoint(kTarget.aLevels[nLevel].im, vIR[i]);
        if(nZMSSD < nBestZMSSD)
        {
            nBest = i;
            nBestZMSSD = nZMSSD;
        }
    }

    if(nBest == -1)   return false;   // Nothing found.

    //  Found a likely candidate along epipolar ray
    Finder.MakeSubPixTemplate();
    Finder.SetSubPixPos(LevelZeroPos(vIR[nBest], nLevel));
    bool bSubPixConverges = Finder.IterateSubPixToConvergence(kTarget,10);
    if(!bSubPixConverges)
        return false;

    // Now triangulate the 3d point...
    Vector<3> v3New;
    v3New = kTarget.se3CfromW.inverse() *
            ReprojectPoint(kSrc.se3CfromW * kTarget.se3CfromW.inverse(),
                           mCamera.UnProject(v2RootPos),
                           mCamera.UnProject(Finder.GetSubPixPos()));

    MapPoint *pNew = new MapPoint;
    pNew->v3WorldPos = v3New;
    pNew->pMMData = new MapMakerData();

    // Patch source stuff:
    pNew->pPatchSourceKF = &kSrc;
    pNew->nSourceLevel = nLevel;
    pNew->v3Normal_NC = makeVector( 0,0,-1);
    pNew->irCenter = irLevelPos;
    pNew->v3Center_NC = unproject(mCamera.UnProject(v2RootPos));
    pNew->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(v2RootPos + vec(ImageRef(nLevelScale,0))));
    pNew->v3OneDownFromCenter_NC  = unproject(mCamera.UnProject(v2RootPos + vec(ImageRef(0,nLevelScale))));

    normalize(pNew->v3Center_NC);
    normalize(pNew->v3OneDownFromCenter_NC);
    normalize(pNew->v3OneRightFromCenter_NC);

    pNew->RefreshPixelVectors();
    
    mMap.vpPoints.push_back(pNew);
    mqNewQueue.push(pNew);
    Measurement m;
    m.Source = Measurement::SRC_ROOT;
    m.v2RootPos = v2RootPos;
    m.nLevel = nLevel;
    m.bSubPix = true;
    kSrc.mMeasurements[pNew] = m;

    m.Source = Measurement::SRC_EPIPOLAR;
    m.v2RootPos = Finder.GetSubPixPos();
    kTarget.mMeasurements[pNew] = m;
    pNew->pMMData->sMeasurementKFs.insert(&kSrc);
    pNew->pMMData->sMeasurementKFs.insert(&kTarget);
    return true;
}
//t的距离最小
double MapMaker::KeyFrameLinearDist(KeyFrame &k1, KeyFrame &k2)
{
  Vector<3> v3KF1_CamPos = k1.se3CfromW.inverse().get_translation();
  Vector<3> v3KF2_CamPos = k2.se3CfromW.inverse().get_translation();
  Vector<3> v3Diff = v3KF2_CamPos - v3KF1_CamPos;
  double dDist = sqrt(v3Diff * v3Diff);
  return dDist;
}

vector<KeyFrame*> MapMaker::NClosestKeyFrames(KeyFrame &k, unsigned int N)
{
  vector<pair<double, KeyFrame* > > vKFandScores;
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    {
      if(mMap.vpKeyFrames[i] == &k)
	continue;
      double dDist = KeyFrameLinearDist(k, *mMap.vpKeyFrames[i]);
      vKFandScores.push_back(make_pair(dDist, mMap.vpKeyFrames[i]));
    }
  if(N > vKFandScores.size())
    N = vKFandScores.size();
  partial_sort(vKFandScores.begin(), vKFandScores.begin() + N, vKFandScores.end());
  
  vector<KeyFrame*> vResult;
  for(unsigned int i=0; i<N; i++)
    vResult.push_back(vKFandScores[i].second);
  return vResult;
}

KeyFrame* MapMaker::ClosestKeyFrame(KeyFrame &k)
{
    double dClosestDist = 9999999999.9;
    int nClosest = -1;
    for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    {
        if(mMap.vpKeyFrames[i] == &k)
            continue;
        double dDist = KeyFrameLinearDist(k, *mMap.vpKeyFrames[i]);
        if(dDist < dClosestDist)
        {
            dClosestDist = dDist;
            nClosest = i;
        }
    }
    assert(nClosest != -1);
    return mMap.vpKeyFrames[nClosest];
}

double MapMaker::DistToNearestKeyFrame(KeyFrame &kCurrent)
{
  KeyFrame *pClosest = ClosestKeyFrame(kCurrent);
  double dDist = KeyFrameLinearDist(kCurrent, *pClosest);
  return dDist;
}

bool MapMaker::NeedNewKeyFrame(KeyFrame &kCurrent)
{
  KeyFrame *pClosest = ClosestKeyFrame(kCurrent);
  double dDist = KeyFrameLinearDist(kCurrent, *pClosest);
  dDist *= (1.0 / kCurrent.dSceneDepthMean);
  
  if(dDist > GV2.GetDouble("MapMaker.MaxKFDistWiggleMult",1.0,SILENT) * mdWiggleScaleDepthNormalized)
    return true;
  return false;
}

// Perform bundle adjustment on all keyframes, all map points
void MapMaker::BundleAdjustAll()
{
  // construct the sets of kfs/points to be adjusted:
  // in this case, all of them
  set<KeyFrame*> sAdj;
  set<KeyFrame*> sFixed;
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    if(mMap.vpKeyFrames[i]->bFixed)
      sFixed.insert(mMap.vpKeyFrames[i]);
    else
      sAdj.insert(mMap.vpKeyFrames[i]);
  
  set<MapPoint*> sMapPoints;
  for(unsigned int i=0; i<mMap.vpPoints.size();i++)
    sMapPoints.insert(mMap.vpPoints[i]);
  
  BundleAdjust(sAdj, sFixed, sMapPoints, false);
}

// Peform a local bundle adjustment which only adjusts
// recently added key-frames
void MapMaker::BundleAdjustRecent()
{
  if(mMap.vpKeyFrames.size() < 8)  
    { // Ignore this unless map is big enough
      mbBundleConverged_Recent = true;
      return;
    }

  // First, make a list of the keyframes we want adjusted in the adjuster.
  // This will be the last keyframe inserted, and its four nearest neighbors
  set<KeyFrame*> sAdjustSet;
  KeyFrame *pkfNewest = mMap.vpKeyFrames.back();
  sAdjustSet.insert(pkfNewest);
  vector<KeyFrame*> vClosest = NClosestKeyFrames(*pkfNewest, 4);
  for(int i=0; i<4; i++)
    if(vClosest[i]->bFixed == false)
      sAdjustSet.insert(vClosest[i]);
  
  // Now we find the set of features which they contain.
  set<MapPoint*> sMapPoints;
  for(set<KeyFrame*>::iterator iter = sAdjustSet.begin();
      iter!=sAdjustSet.end();
      iter++)
    {
      map<MapPoint*,Measurement> &mKFMeas = (*iter)->mMeasurements;
      for(meas_it jiter = mKFMeas.begin(); jiter!= mKFMeas.end(); jiter++)
	sMapPoints.insert(jiter->first);
    };
  
  // Finally, add all keyframes which measure above points as fixed keyframes
  set<KeyFrame*> sFixedSet;
  for(vector<KeyFrame*>::iterator it = mMap.vpKeyFrames.begin(); it!=mMap.vpKeyFrames.end(); it++)
    {
      if(sAdjustSet.count(*it))
	continue;
      bool bInclude = false;
      for(meas_it jiter = (*it)->mMeasurements.begin(); jiter!= (*it)->mMeasurements.end(); jiter++)
	if(sMapPoints.count(jiter->first))
	  {
	    bInclude = true;
	    break;
	  }
      if(bInclude)
	sFixedSet.insert(*it);
    }
  
  BundleAdjust(sAdjustSet, sFixedSet, sMapPoints, true);
}

// Common bundle adjustment code. This creates a bundle-adjust instance, populates it, and runs it.
void MapMaker::BundleAdjust(set<KeyFrame*> sAdjustSet, set<KeyFrame*> sFixedSet, set<MapPoint*> sMapPoints, bool bRecent)
{
  Bundle b(mCamera);   // Our bundle adjuster
  mbBundleRunning = true;
  mbBundleRunningIsRecent = bRecent;
  
  // The bundle adjuster does different accounting of keyframes and map points;
  // Translation maps are stored:
  map<MapPoint*, int> mPoint_BundleID;
  map<int, MapPoint*> mBundleID_Point;
  map<KeyFrame*, int> mView_BundleID;
  map<int, KeyFrame*> mBundleID_View;
  
  // Add the keyframes' poses to the bundle adjuster. Two parts: first nonfixed, then fixed.
  for(set<KeyFrame*>::iterator it = sAdjustSet.begin(); it!= sAdjustSet.end(); it++)
    {
      int nBundleID = b.AddCamera((*it)->se3CfromW, (*it)->bFixed);
      mView_BundleID[*it] = nBundleID;
      mBundleID_View[nBundleID] = *it;
    }
  for(set<KeyFrame*>::iterator it = sFixedSet.begin(); it!= sFixedSet.end(); it++)
    {
      int nBundleID = b.AddCamera((*it)->se3CfromW, true);
      mView_BundleID[*it] = nBundleID;
      mBundleID_View[nBundleID] = *it;
    }
  
  // Add the points' 3D position
  for(set<MapPoint*>::iterator it = sMapPoints.begin(); it!=sMapPoints.end(); it++)
    {
      int nBundleID = b.AddPoint((*it)->v3WorldPos);
      mPoint_BundleID[*it] = nBundleID;
      mBundleID_Point[nBundleID] = *it;
    }
  
  // Add the relevant point-in-keyframe measurements
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    {
      if(mView_BundleID.count(mMap.vpKeyFrames[i]) == 0)
	continue;
      
      int nKF_BundleID = mView_BundleID[mMap.vpKeyFrames[i]];
      for(meas_it it= mMap.vpKeyFrames[i]->mMeasurements.begin();
	  it!= mMap.vpKeyFrames[i]->mMeasurements.end();
	  it++)
	{
	  if(mPoint_BundleID.count(it->first) == 0)
	    continue;
	  int nPoint_BundleID = mPoint_BundleID[it->first];
	  b.AddMeas(nKF_BundleID, nPoint_BundleID, it->second.v2RootPos, LevelScale(it->second.nLevel) * LevelScale(it->second.nLevel));
	}
    }
  
  // Run the bundle adjuster. This returns the number of successful iterations
  int nAccepted = b.Compute(&mbBundleAbortRequested);
  
  if(nAccepted < 0)
    {
      // Crap: - LM Ran into a serious problem!
      // This is probably because the initial stereo was messed up.
      // Get rid of this map and start again! 
      cout << "!! MapMaker: Cholesky failure in bundle adjust. " << endl
	   << "   The map is probably corrupt: Ditching the map. " << endl;
      mbResetRequested = true;
      return;
    }

  // Bundle adjustment did some updates, apply these to the map
  if(nAccepted > 0)
    {
      
      for(map<MapPoint*,int>::iterator itr = mPoint_BundleID.begin();
	  itr!=mPoint_BundleID.end();
	  itr++)
	itr->first->v3WorldPos = b.GetPoint(itr->second);
      
      for(map<KeyFrame*,int>::iterator itr = mView_BundleID.begin();
	  itr!=mView_BundleID.end();
	  itr++)
	itr->first->se3CfromW = b.GetCamera(itr->second);
      if(bRecent)
	mbBundleConverged_Recent = false;
      mbBundleConverged_Full = false;
    };
  
  if(b.Converged())
    {
      mbBundleConverged_Recent = true;
      if(!bRecent)
	mbBundleConverged_Full = true;
    }
  
  mbBundleRunning = false;
  mbBundleAbortRequested = false;
  
  // Handle outlier measurements:
  vector<pair<int,int> > vOutliers_PC_pair = b.GetOutlierMeasurements();
  for(unsigned int i=0; i<vOutliers_PC_pair.size(); i++)
    {
      MapPoint *pp = mBundleID_Point[vOutliers_PC_pair[i].first];
      KeyFrame *pk = mBundleID_View[vOutliers_PC_pair[i].second];
      Measurement &m = pk->mMeasurements[pp];
      if(pp->pMMData->GoodMeasCount() <= 2 || m.Source == Measurement::SRC_ROOT)   // Is the original source kf considered an outlier? That's bad.
	pp->bBad = true;
      else
	{
	  // Do we retry it? Depends where it came from!!
	  if(m.Source == Measurement::SRC_TRACKER || m.Source == Measurement::SRC_EPIPOLAR)
	    mvFailureQueue.push_back(pair<KeyFrame*,MapPoint*>(pk,pp));
	  else
	    pp->pMMData->sNeverRetryKFs.insert(pk);
	  pk->mMeasurements.erase(pp);
	  pp->pMMData->sMeasurementKFs.erase(pk);
	}
    }
}

// Mapmaker's try-to-find-a-point-in-a-keyframe code. This is used to update
// data association if a bad measurement was detected, or if a point
// was never searched for in a keyframe in the first place. This operates
// much like the tracker! So most of the code looks just like in 
// TrackerData.h.
bool MapMaker::ReFind_Common(KeyFrame &k, MapPoint &p)
{
  // abort if either a measurement is already in the map, or we've
  // decided that this point-kf combo is beyond redemption
  if(p.pMMData->sMeasurementKFs.count(&k)
     || p.pMMData->sNeverRetryKFs.count(&k))
    return false;
  
  static PatchFinder Finder;
  Vector<3> v3Cam = k.se3CfromW*p.v3WorldPos;
  if(v3Cam[2] < 0.001)
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }
  Vector<2> v2ImPlane = project(v3Cam);
  if(v2ImPlane* v2ImPlane > mCamera.LargestRadiusInImage() * mCamera.LargestRadiusInImage())
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }
  
  Vector<2> v2Image = mCamera.Project(v2ImPlane);
  if(mCamera.Invalid())
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }

  ImageRef irImageSize = k.aLevels[0].im.size();
  if(v2Image[0] < 0 || v2Image[1] < 0 || v2Image[0] > irImageSize[0] || v2Image[1] > irImageSize[1])
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }
  
  Matrix<2> m2CamDerivs = mCamera.GetProjectionDerivs();
  Finder.MakeTemplateCoarse(p, k.se3CfromW, m2CamDerivs);
  
  if(Finder.TemplateBad())
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }
  
  bool bFound = Finder.FindPatchCoarse(ir(v2Image), k, 4);  // Very tight search radius!
  if(!bFound)
    {
      p.pMMData->sNeverRetryKFs.insert(&k);
      return false;
    }
  
  // If we found something, generate a measurement struct and put it in the map
  Measurement m;
  m.nLevel = Finder.GetLevel();
  m.Source = Measurement::SRC_REFIND;
  
  if(Finder.GetLevel() > 0)
    {
      Finder.MakeSubPixTemplate();
      Finder.IterateSubPixToConvergence(k,8);
      m.v2RootPos = Finder.GetSubPixPos();
      m.bSubPix = true;
    }
  else
    {
      m.v2RootPos = Finder.GetCoarsePosAsVector();
      m.bSubPix = false;
    };
  
  if(k.mMeasurements.count(&p))
    {
      assert(0); // This should never happen, we checked for this at the start.
    }
  k.mMeasurements[&p] = m;
  p.pMMData->sMeasurementKFs.insert(&k);
  return true;
}

// A general data-association update for a single keyframe
// Do this on a new key-frame when it's passed in by the tracker
int MapMaker::ReFindInSingleKeyFrame(KeyFrame &k)
{
  vector<MapPoint*> vToFind;
  for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
    vToFind.push_back(mMap.vpPoints[i]);
  
  int nFoundNow = 0;
  for(unsigned int i=0; i<vToFind.size(); i++)
    if(ReFind_Common(k,*vToFind[i]))
      nFoundNow++;

  return nFoundNow;
};

// When new map points are generated, they're only created from a stereo pair
// this tries to make additional measurements in other KFs which they might
// be in.
void MapMaker::ReFindNewlyMade()
{
  if(mqNewQueue.empty())
    return;
  int nFound = 0;
  int nBad = 0;
  while(!mqNewQueue.empty() && mvpKeyFrameQueue.size() == 0)
    {
      MapPoint* pNew = mqNewQueue.front();
      mqNewQueue.pop();
      if(pNew->bBad)
	{
	  nBad++;
	  continue;
	}
      for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
	if(ReFind_Common(*mMap.vpKeyFrames[i], *pNew))
	  nFound++;
    }
};

// Dud measurements get a second chance.
void MapMaker::ReFindFromFailureQueue()
{
  if(mvFailureQueue.size() == 0)
    return;
  sort(mvFailureQueue.begin(), mvFailureQueue.end());
  vector<pair<KeyFrame*, MapPoint*> >::iterator it;
  int nFound=0;
  for(it = mvFailureQueue.begin(); it!=mvFailureQueue.end(); it++)
    if(ReFind_Common(*it->first, *it->second))
      nFound++;
  
  mvFailureQueue.erase(mvFailureQueue.begin(), it);
};

// Is the tracker's camera pose in cloud-cuckoo land?
bool MapMaker::IsDistanceToNearestKeyFrameExcessive(KeyFrame &kCurrent)
{
  return DistToNearestKeyFrame(kCurrent) > mdWiggleScale * 10.0;
}

// Find a dominant plane in the map, find an SE3<> to put it as the z=0 plane
SE3<> MapMaker::CalcPlaneAligner()
{
  unsigned int nPoints = mMap.vpPoints.size();
  if(nPoints < 10)
    {
      cout << "  MapMaker: CalcPlane: too few points to calc plane." << endl;
      return SE3<>();
    };
  
  int nRansacs = GV2.GetInt("MapMaker.PlaneAlignerRansacs", 100, HIDDEN|SILENT);
  Vector<3> v3BestMean;
  Vector<3> v3BestNormal;
  double dBestDistSquared = 9999999999999999.9;
  
  for(int i=0; i<nRansacs; i++)
    {
      int nA = rand()%nPoints;
      int nB = nA;
      int nC = nA;
      while(nB == nA)
	nB = rand()%nPoints;
      while(nC == nA || nC==nB)
	nC = rand()%nPoints;
      
      Vector<3> v3Mean = 0.33333333 * (mMap.vpPoints[nA]->v3WorldPos + 
				       mMap.vpPoints[nB]->v3WorldPos + 
				       mMap.vpPoints[nC]->v3WorldPos);
      
      Vector<3> v3CA = mMap.vpPoints[nC]->v3WorldPos  - mMap.vpPoints[nA]->v3WorldPos;
      Vector<3> v3BA = mMap.vpPoints[nB]->v3WorldPos  - mMap.vpPoints[nA]->v3WorldPos;
      Vector<3> v3Normal = v3CA ^ v3BA;
      if(v3Normal * v3Normal  == 0)
	continue;
      normalize(v3Normal);
      
      double dSumError = 0.0;
      for(unsigned int i=0; i<nPoints; i++)
	{
	  Vector<3> v3Diff = mMap.vpPoints[i]->v3WorldPos - v3Mean;
	  double dDistSq = v3Diff * v3Diff;
	  if(dDistSq == 0.0)
	    continue;
	  double dNormDist = fabs(v3Diff * v3Normal);
	  
	  if(dNormDist > 0.05)
	    dNormDist = 0.05;
	  dSumError += dNormDist;
	}
      if(dSumError < dBestDistSquared)
	{
	  dBestDistSquared = dSumError;
	  v3BestMean = v3Mean;
	  v3BestNormal = v3Normal;
	}
    }
  
  // Done the ransacs, now collect the supposed inlier set
  vector<Vector<3> > vv3Inliers;
  for(unsigned int i=0; i<nPoints; i++)
    {
      Vector<3> v3Diff = mMap.vpPoints[i]->v3WorldPos - v3BestMean;
      double dDistSq = v3Diff * v3Diff;
      if(dDistSq == 0.0)
	continue;
      double dNormDist = fabs(v3Diff * v3BestNormal);
      if(dNormDist < 0.05)
	vv3Inliers.push_back(mMap.vpPoints[i]->v3WorldPos);
    }
  
  // With these inliers, calculate mean and cov
  Vector<3> v3MeanOfInliers = Zeros;
  for(unsigned int i=0; i<vv3Inliers.size(); i++)
    v3MeanOfInliers+=vv3Inliers[i];
  v3MeanOfInliers *= (1.0 / vv3Inliers.size());
  
  Matrix<3> m3Cov = Zeros;
  for(unsigned int i=0; i<vv3Inliers.size(); i++)
    {
      Vector<3> v3Diff = vv3Inliers[i] - v3MeanOfInliers;
      m3Cov += v3Diff.as_col() * v3Diff.as_row();
    };
  
  // Find the principal component with the minimal variance: this is the plane normal
  SymEigen<3> sym(m3Cov);
  Vector<3> v3Normal = sym.get_evectors()[0];
  
  // Use the version of the normal which points towards the cam center
  if(v3Normal[2] > 0)
    v3Normal *= -1.0;
  
  Matrix<3> m3Rot = Identity;
  m3Rot[2] = v3Normal;
  m3Rot[0] = m3Rot[0] - (v3Normal * (m3Rot[0] * v3Normal));
  normalize(m3Rot[0]);
  m3Rot[1] = m3Rot[2] ^ m3Rot[0];
  
  SE3<> se3Aligner;
  se3Aligner.get_rotation() = m3Rot;
  Vector<3> v3RMean = se3Aligner * v3MeanOfInliers;
  se3Aligner.get_translation() = -v3RMean;
  
  return se3Aligner;
}

// Calculates the depth(z-) distribution of map points visible in a keyframe
// This function is only used for the first two keyframes - all others
// get this filled in by the tracker
void MapMaker::RefreshSceneDepth(KeyFrame *pKF)
{
  double dSumDepth = 0.0;
  double dSumDepthSquared = 0.0;
  int nMeas = 0;
  for(meas_it it = pKF->mMeasurements.begin(); it!=pKF->mMeasurements.end(); it++)
    {
      MapPoint &point = *it->first;
      Vector<3> v3PosK = pKF->se3CfromW * point.v3WorldPos;
      dSumDepth += v3PosK[2];
      dSumDepthSquared += v3PosK[2] * v3PosK[2];
      nMeas++;
    }
 
  assert(nMeas > 2); // If not then something is seriously wrong with this KF!!
  pKF->dSceneDepthMean = dSumDepth / nMeas;
  pKF->dSceneDepthSigma = sqrt((dSumDepthSquared / nMeas) - (pKF->dSceneDepthMean) * (pKF->dSceneDepthMean));
}

void MapMaker::GUICommandCallBack(void* ptr, string sCommand, string sParams)
{
  Command c;
  c.sCommand = sCommand;
  c.sParams = sParams;
  ((MapMaker*) ptr)->mvQueuedCommands.push_back(c);
}

void MapMaker::GUICommandHandler(string sCommand, string sParams)  // Called by the callback func..
{
  if(sCommand=="SaveMap")
    {
      cout << "  MapMaker: Saving the map.... " << endl;
      ofstream ofs("map.dump");
      for(unsigned int i=0; i<mMap.vpPoints.size(); i++)
	{
	  ofs << mMap.vpPoints[i]->v3WorldPos << "  ";
	  ofs << mMap.vpPoints[i]->nSourceLevel << endl;
	}
      ofs.close();
      
      for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
	{
	  ostringstream ost1;
	  ost1 << "keyframes/" << i << ".jpg";
//	  img_save(mMap.vpKeyFrames[i]->aLevels[0].im, ost1.str());
	  
	  ostringstream ost2;
	  ost2 << "keyframes/" << i << ".info";
	  ofstream ofs2;
	  ofs2.open(ost2.str().c_str());
	  ofs2 << mMap.vpKeyFrames[i]->se3CfromW << endl;
	  ofs2.close();
	}
      cout << "  ... done saving map." << endl;
      return;
    }
  
  cout << "! MapMaker::GUICommandHandler: unhandled command "<< sCommand << endl;
  exit(1);
}; 


bool MapMaker::InitFromStereoForFast(std::list<Trail> mlTrails, cv::Mat &src, KeyFrame &kF,
                              KeyFrame &kS,
                              SE3<> &se3TrackerPose)
{




    mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

    mCamera.SetImageSize(kF.aLevels[0].im.size());

    KeyFrame *pkFirst = new KeyFrame();
    KeyFrame *pkSecond = new KeyFrame();
    *pkFirst = kF;
    *pkSecond = kF;

    pkFirst->bFixed = true;
    pkFirst->se3CfromW = SE3<>();

    pkSecond->bFixed = true;
    pkSecond->se3CfromW = SE3<>();

    //compute vanishing point
    mvVanishingPoints.clear();
    int numVps = 0;
    DetectVanishingPoints(src, mvVanishingPoints);
    numVps = mvVanishingPoints.size();

    cout<<"numVps: "<<numVps<<endl;

    double maxDist=0, minDist=100000;
    if (numVps>0) {
        hasNewDetectedVPs = true;
        //对每一个特征点去计算该特征点到灭点的距离之和
        std::list<Trail>::iterator it;
        for (it = mlTrails.begin(); it != mlTrails.end(); it++)
        {
            double dist = 0;

            for (int vi=0; vi<numVps; vi++)
            {
                dist += sqrt((it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) * (it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) + (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)) * (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)));
            }
            if (dist>maxDist){
                maxDist = dist;
            }
            if (dist<minDist) {
                minDist = dist;
            }
        }
        //cout << minDist << "   " << maxDist << endl;
    }



    // Create MapPoints and asscoiate to keyframes
    std::list<Trail>::iterator it;
    for (it = mlTrails.begin(); it != mlTrails.end(); it++)
    {

        MapPoint *p = new MapPoint();

        // Patch source stuff:
        p->pPatchSourceKF = pkFirst;
        p->nSourceLevel = 0;
        p->v3Normal_NC = makeVector( 0,0,-1);
        p->irCenter = it->irInitialPos;
        p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
        p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
        p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
        normalize(p->v3Center_NC);
        normalize(p->v3OneDownFromCenter_NC);
        normalize(p->v3OneRightFromCenter_NC);
        p->RefreshPixelVectors();


        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> normal(1,0.125);
        double z_depth;
        if (numVps==0) {
            z_depth = normal(gen);
        }
        else {
            double dist = 0;
            for (int vi=0; vi<numVps; vi++){
                dist += sqrt((it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) * (it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) + (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)) * (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)));
            }
            z_depth = (maxDist-dist)/(maxDist-minDist)+0.5;//+normal(gen)/10;
        }
        //std::cout<<z_depth<<std::endl;
//        float p_x = (it->irInitialPos.x-mK.at<float>(0,2)) / mK.at<float>(0,0);
//        float p_y = (it->irInitialPos.y-mK.at<float>(1,2)) / mK.at<float>(1,1);
        double p_x = it->irInitialPos.x;
        double p_y = it->irInitialPos.y;
        //double d[] = {p_x, p_y, z_depth};
        Vector<3> vP;
        vP[0] = p_x;vP[1] = p_y;vP[2] = z_depth;

        p->v3WorldPos = vP;
        // Not behind map? Good, then add to map.
        p->pMMData = new MapMakerData();
        mMap.vpPoints.push_back(p);

        // Construct first two measurements and insert into relevant DBs:
        Measurement mFirst;
        mFirst.nLevel = 0;
        mFirst.Source = Measurement::SRC_ROOT;
        mFirst.v2RootPos = vec(it->irInitialPos);
        mFirst.bSubPix = true;
        pkFirst->mMeasurements[p] = mFirst;
        p->pMMData->sMeasurementKFs.insert(pkFirst);

        Measurement mSecond;
        mSecond.nLevel = 0;
        mSecond.Source = Measurement::SRC_TRAIL;
        mSecond.v2RootPos = vec(it->irInitialPos);
        mSecond.bSubPix = true;
        pkSecond->mMeasurements[p] = mSecond;
        p->pMMData->sMeasurementKFs.insert(pkSecond);

    }
    mMap.vpKeyFrames.push_back(pkFirst);
    mMap.vpKeyFrames.push_back(pkSecond);
    pkFirst->MakeKeyFrame_Rest();
    pkSecond->MakeKeyFrame_Rest();


//    // Estimate the feature depth distribution in the first two key-frames
//    // (Needed for epipolar search)
//    RefreshSceneDepth(pkFirst);
//    RefreshSceneDepth(pkSecond);
//    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;


//    AddSomeMapPoints(0);
//    AddSomeMapPoints(3);
//    AddSomeMapPoints(1);
//    AddSomeMapPoints(2);
    mMap.bGood = true;
    se3TrackerPose = pkFirst->se3CfromW;
    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;
    return true;




}

void MapMaker::LineDetect( cv::Mat image, double thLength, std::vector<std::vector<cv::Point> > &lines )
{

    image_double imageLSD = new_image_double( image.cols, image.rows );
    unsigned char* im_src = (unsigned char*) image.data;

//     cv::cvtColor(image, image, CV_GRAY2BGR);

    int xsize = image.cols;
    int ysize = image.rows;
    for ( int y = 0; y < ysize; ++y )
    {
    for ( int x = 0; x < xsize; ++x )
        {
        imageLSD->data[y * xsize + x] = im_src[y * xsize + x];
        }
    }

    ntuple_list linesLSD = lsd( imageLSD );
    free_image_double( imageLSD );

    vector<cv::Point> aux;
    int nLines = linesLSD->size;
    int dim = linesLSD->dim;
    for ( int i = 0; i < nLines; ++i )
    {
        double x1 = floor(linesLSD->values[i * dim + 0]);
        double y1 = floor(linesLSD->values[i * dim + 1]);
        double x2 = floor(linesLSD->values[i * dim + 2]);
        double y2 = floor(linesLSD->values[i * dim + 3]);

        double l = sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) );
        if ( l > thLength )
        {
        aux.clear();
            aux.push_back(cv::Point(x1,y1));
            aux.push_back(cv::Point(x2,y2));

            lines.push_back( aux );
// 	    cv::line(image, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0, 0, 255), 1);
        }
    }

    free_ntuple_list(linesLSD);
//     cv::imwrite("/home/alan/Projects/ORB_SLAM2-master/Examples/Monocular/results/lines.png", image);
}

void MapMaker::DetectVanishingPoints(cv::Mat src, std::vector<cv::Mat> &vps)
{
    cv::Mat img = src.clone();
    mvLineSegments.clear();
    vector<cv::Point> aux;
    LineDetect(img,40,mvLineSegments);
    std::cout<<"number of linesegments "<<mvLineSegments.size()<<std::endl;
    std::vector<int> numInliers;
    int w1 = src.cols;
    int h1 = src.rows;

    mvLineSegmentsClusters.clear();
    int numVps=3;

    MSAC msac;
    msac.init(MODE_NIETO, cv::Size(w1, h1), false);
        // Call msac function for multiple vanishing point estimation
    msac.multipleVPEstimation(mvLineSegments, mvLineSegmentsClusters, numInliers, vps, numVps);

//     cv::cvtColor(img, img, CV_GRAY2BGR);
//
//     numVps = vps.size();


// //     clusterlies DRAW
//     for(int num=0; num<mvLineSegmentsClusters.size(); ++num)
//     {
//       for(int i=0; i<mvLineSegmentsClusters[num].size(); ++i)
//       {
// 	if(num == 0)
// 	  cv::line(img, cv::Point(mvLineSegmentsClusters[num][i][0].x,mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(mvLineSegmentsClusters[num][i][1].x, mvLineSegmentsClusters[num][i][1].y), cv::Scalar(0, 0, 255), 1);
// 	else if(num == 1)
// 	  cv::line(img, cv::Point(mvLineSegmentsClusters[num][i][0].x,mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(mvLineSegmentsClusters[num][i][1].x, mvLineSegmentsClusters[num][i][1].y), cv::Scalar(0, 255, 0), 1);
// 	else
// 	  cv::line(img, cv::Point(mvLineSegmentsClusters[num][i][0].x,mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(mvLineSegmentsClusters[num][i][1].x, mvLineSegmentsClusters[num][i][1].y), cv::Scalar(255, 0, 0), 1);
//       }
//     }
//
//     cv::imwrite("/home/alan/Projects/ORB_SLAM2-master/Examples/Monocular/results/clusterlines.png", img);
//
//     int min_x = 0, max_x = w1;
//     int min_y = 0, max_y = h1;
//     int pad = 200;
//     cout<<"VanishingPoints: "<<numVps<<endl;
//     for(int i=0; i<numVps; ++i)
//     {
// 	cout<<mvVanishingPoints[i].at<float>(0,0)<<" "<<mvVanishingPoints[i].at<float>(0,1)<<endl;
// 	if(min_x>mvVanishingPoints[i].at<float>(0,0)) min_x = mvVanishingPoints[i].at<float>(0,0);
// 	if(max_x<mvVanishingPoints[i].at<float>(0,0)) max_x = mvVanishingPoints[i].at<float>(0,0);
// 	if(min_y>mvVanishingPoints[i].at<float>(0,1)) min_y = mvVanishingPoints[i].at<float>(0,1);
// 	if(max_y<mvVanishingPoints[i].at<float>(0,1)) max_y = mvVanishingPoints[i].at<float>(0,1);
//     }
//
//     cout<<"MM: "<<min_x<<" "<<max_x<<" "<<min_y<<" "<<max_y<<endl;
//
//     cv::Mat vpMat(cv::Size(max_x-min_x+2*pad, max_y-min_y+2*pad), CV_8UC3, cv::Scalar(0,0,0));
//     cout<<"WH: "<<max_x-min_x+2*pad<<" "<<max_y-min_y+2*pad<<endl;
//     cv::Rect rect(pad-min_x, pad-min_y, w1, h1);
//     cout<<"Start: "<<pad-min_x<<" "<<pad-min_y<<endl;
//     cv::Mat roiImage = vpMat(rect);
//     img.copyTo(roiImage);
//
//     cv::imwrite("/home/alan/Projects/ORB_SLAM2-master/Examples/Monocular/results/vpMat.png", vpMat);
//
//     for(int i=0; i<numVps; ++i)
//     {
// 	  cv::circle(vpMat, cv::Point(pad-min_x+mvVanishingPoints[i].at<float>(0,0), pad-min_y+mvVanishingPoints[i].at<float>(0,1)), 10,
// 				      cv::Scalar(255,255,255), 10);
//     }
//
//     for(int num=0; num<numVps; ++num)
//     {
//       for(int i=0; i<mvLineSegmentsClusters[num].size(); ++i)
//       {
// 	if(num == 0){
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)),
// 		 cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][1].x, pad-min_y+mvLineSegmentsClusters[num][i][1].y), cv::Scalar(0, 0, 255), 1);
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][0].x,pad-min_y+mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)), cv::Scalar(0, 0, 255), 1);
// 	}
// 	else if(num == 1)
// 	{
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)),
// 		 cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][1].x, pad-min_y+mvLineSegmentsClusters[num][i][1].y), cv::Scalar(0, 255, 0), 1);
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][0].x,pad-min_y+mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)), cv::Scalar(0, 255, 0), 1);
// 	}
// 	else if(num ==2)
// 	{
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)),
// 		 cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][1].x, pad-min_y+mvLineSegmentsClusters[num][i][1].y), cv::Scalar(255, 0, 0), 1);
// 	  cv::line(vpMat, cv::Point(pad-min_x+mvLineSegmentsClusters[num][i][0].x,pad-min_y+mvLineSegmentsClusters[num][i][0].y),
// 		 cv::Point(pad-min_x+mvVanishingPoints[num].at<float>(0,0), pad-min_y+mvVanishingPoints[num].at<float>(0,1)), cv::Scalar(255, 0, 0), 1);
// 	}
//       }
//     }
//
//     cv::imwrite("/home/alan/Projects/ORB_SLAM2-master/Examples/Monocular/results/vpMatFinal.png", vpMat);

}



//替换深度
bool MapMaker::InitFromStereo2(std::list<Trail> mlTrails, cv::Mat src, //KeyFrame &kF,
                              KeyFrame &kS,
                              //vector<pair<ImageRef, ImageRef> > &vTrailMatches,
                              SE3<> &se3TrackerPose)
{

//    cout << "kS.aLevels[0].im.size() " << kS.aLevels[0].im.size() << endl;
//    cout << "kF.aLevels[0].im.size() " << kF.aLevels[0].im.size() << endl;

    double t = (double)cvGetTickCount();
    //compute vanishing point
    mvVanishingPoints.clear();
    int numVps = 0;
    DetectVanishingPoints(src, mvVanishingPoints);
    numVps = mvVanishingPoints.size();

    cout<<"numVps: "<<numVps<<endl;
    t = (double)cvGetTickCount() - t;
    //cout << "t " << t << endl;
    t = t/(cvGetTickFrequency()*1000000.0);
    cout << "tttt " << t << endl;








    mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

    mCamera.SetImageSize(kS.aLevels[0].im.size());


//    vector<HomographyMatch> vMatches;
//    for(unsigned int i=0; i<vTrailMatches.size(); i++)
//    {
//        HomographyMatch m;
//        m.v2CamPlaneFirst = mCamera.UnProject(vTrailMatches[i].first);
//        m.v2CamPlaneSecond = mCamera.UnProject(vTrailMatches[i].second);
//        m.m2PixelProjectionJac = mCamera.GetProjectionDerivs();
//        vMatches.push_back(m);
//    }

//    SE3<> se3;
//    bool bGood;
//    HomographyInit HomographyInit;
//    bGood = HomographyInit.Compute(vMatches, 5.0, se3);
//    if(!bGood)
//    {
//        cout << "  Could not init from stereo pair, try again." << endl;
//        return false;
//    }

//    // Check that the initialiser estimated a non-zero baseline
//    double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
//    if(dTransMagn == 0)
//    {
//        cout << "  Estimated zero baseline from stereo pair, try again." << endl;
//        return false;
//    }
//    // change the scale of the map so the second camera is wiggleScale away from the first
//    se3.get_translation() *= mdWiggleScale/dTransMagn;


    KeyFrame *pkFirst = new KeyFrame();
    KeyFrame *pkSecond = new KeyFrame();
    *pkFirst = kS;
    *pkSecond = kS;

    pkFirst->bFixed = true;
    pkFirst->se3CfromW = SE3<>();

    pkSecond->bFixed = true;
    pkSecond->se3CfromW = SE3<>();

    // Construct map from the stereo matches.
//    PatchFinder finder;

//    for (unsigned int i=0; i<vMatches.size(); i++) {

//    }
    double maxDist=0, minDist=100000;
    if (numVps>0) {
        hasNewDetectedVPs = true;
        //对每一个特征点去计算该特征点到灭点的距离之和
        std::list<Trail>::iterator it;
        for (it = mlTrails.begin(); it != mlTrails.end(); it++)
        {
            double dist = 0;

            for (int vi=0; vi<numVps; vi++)
            {
                dist += sqrt((it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) * (it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) + (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)) * (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)));
            }
            if (dist>maxDist){
                maxDist = dist;
            }
            if (dist<minDist) {
                minDist = dist;
            }
        }
//        cout << minDist << "   " << maxDist << endl;
    }

    // Create MapPoints and asscoiate to keyframes
    std::list<Trail>::iterator it;
    for (it = mlTrails.begin(); it != mlTrails.end(); it++)
    {
        MapPoint *p = new MapPoint();

        // Patch source stuff:
        p->pPatchSourceKF = pkFirst;
        p->nSourceLevel = 0;
        p->v3Normal_NC = makeVector( 0,0,-1);
        p->irCenter = it->irCurrentPos;
        p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
        p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
        p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
        normalize(p->v3Center_NC);
        normalize(p->v3OneDownFromCenter_NC);
        normalize(p->v3OneRightFromCenter_NC);
        p->RefreshPixelVectors();

        // Do sub-pixel alignment on the second image
//        finder.MakeTemplateCoarseNoWarp(*p);
//        finder.MakeSubPixTemplate();
//        finder.SetSubPixPos(vec(vTrailMatches[i].second));
//        bool bGood = finder.IterateSubPixToConvergence(*pkSecond,10);
//        if(!bGood)
//        {
//            delete p; continue;
//        }

        // Triangulate point:
        //Vector<2> v2SecondPos = finder.GetSubPixPos();
//        p->v3WorldPos = ReprojectPoint(se3, mCamera.UnProject(v2SecondPos), vMatches[i].v2CamPlaneFirst);
//        if(p->v3WorldPos[2] < 0.0)
//        {
//            delete p; continue;
//        }


        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> normal(1,0.125);
        double z_depth;
        if (numVps==0) {
            z_depth = normal(gen);
        }
        else {
            double dist = 0;
            for (int vi=0; vi<numVps; vi++){
                dist += sqrt((it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) * (it->irInitialPos.x-mvVanishingPoints[vi].at<float>(0,0)) + (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)) * (it->irInitialPos.y-mvVanishingPoints[vi].at<float>(0,1)));
            }
            z_depth = (maxDist-dist)/(maxDist-minDist)+0.5;//+normal(gen)/10;
        }

//        p->v3WorldPos[0] = (it->irInitialPos.x-200)/1000.0;
//        p->v3WorldPos[1] = (it->irInitialPos.y-200)/1000.0;
//        p->v3WorldPos[0] = (it->irInitialPos.x-200)/1000.0;
//        p->v3WorldPos[1] = (it->irInitialPos.y-200)/1000.0;
        p->v3WorldPos[0] = (mCamera.UnProject(it->irCurrentPos))[0];
        p->v3WorldPos[1] = (mCamera.UnProject(it->irCurrentPos))[1];
        p->v3WorldPos[2] = (z_depth);
        //cout << p->v3WorldPos[0] << " " << p->v3WorldPos[1] << " " << z_depth << endl;

        // Not behind map? Good, then add to map.
        p->pMMData = new MapMakerData();
        mMap.vpPoints.push_back(p);

        // Construct first two measurements and insert into relevant DBs:
        Measurement mFirst;
        mFirst.nLevel = 0;
        mFirst.Source = Measurement::SRC_ROOT;
        mFirst.v2RootPos = vec(it->irInitialPos);
        mFirst.bSubPix = true;
        pkFirst->mMeasurements[p] = mFirst;
        p->pMMData->sMeasurementKFs.insert(pkFirst);

        Measurement mSecond;
        mSecond.nLevel = 0;
        mSecond.Source = Measurement::SRC_TRAIL;
        mSecond.v2RootPos = vec(it->irCurrentPos);
        mSecond.bSubPix = true;
        pkSecond->mMeasurements[p] = mSecond;
        p->pMMData->sMeasurementKFs.insert(pkSecond);
    }


    mMap.vpKeyFrames.push_back(pkFirst);
    mMap.vpKeyFrames.push_back(pkSecond);
    pkFirst->MakeKeyFrame_Rest();
    pkSecond->MakeKeyFrame_Rest();

    for(int i=0; i<5; i++)
        BundleAdjustAll();

    // Estimate the feature depth distribution in the first two key-frames
    // (Needed for epipolar search)
//    RefreshSceneDepth(pkFirst);
//    RefreshSceneDepth(pkSecond);
//    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;

    cout << "  MapMaker0: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    std::vector<CVD::ImageRef> vMapPointsCandidates;//存储Candidates的临时变量
    AddSomeMapPointsForFast (vMapPointsCandidates, 0);
    AddSomeMapPointsForFast (vMapPointsCandidates, 3);
    AddSomeMapPointsForFast (vMapPointsCandidates, 1);
    AddSomeMapPointsForFast (vMapPointsCandidates, 2);
    cout << "vMapPointsCandidates: " << vMapPointsCandidates.size() << endl;

    for (int addPointsNum = 0; addPointsNum < vMapPointsCandidates.size(); addPointsNum++) {
        MapPoint *p = new MapPoint();

        // Patch source stuff:
        p->pPatchSourceKF = pkFirst;
        p->nSourceLevel = 0;
        p->v3Normal_NC = makeVector( 0,0,-1);
        p->irCenter = vMapPointsCandidates[addPointsNum];
        p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
        p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
        p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
        normalize(p->v3Center_NC);
        normalize(p->v3OneDownFromCenter_NC);
        normalize(p->v3OneRightFromCenter_NC);
        p->RefreshPixelVectors();

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> normal(1,0.125);
        double z_depth;
        if (numVps==0) {
            z_depth = normal(gen);
        }
        else {
            double dist = 0;
            for (int vi=0; vi<numVps; vi++){
                dist += sqrt(((vMapPointsCandidates[addPointsNum]).x-mvVanishingPoints[vi].at<float>(0,0)) * ((vMapPointsCandidates[addPointsNum]).x-mvVanishingPoints[vi].at<float>(0,0)) + ((vMapPointsCandidates[addPointsNum]).y-mvVanishingPoints[vi].at<float>(0,1)) * ((vMapPointsCandidates[addPointsNum]).y-mvVanishingPoints[vi].at<float>(0,1)));
            }
            z_depth = (maxDist-dist)/(maxDist-minDist)+0.5;//+normal(gen)/10;
            //cout << "z_depth" << z_depth << endl;
        }

        p->v3WorldPos[0] = (mCamera.UnProject(vMapPointsCandidates[addPointsNum]))[0];
        p->v3WorldPos[1] = (mCamera.UnProject(vMapPointsCandidates[addPointsNum]))[1];
        p->v3WorldPos[2] = (z_depth);
        //cout << p->v3WorldPos[0] << " " << p->v3WorldPos[1] << " " << z_depth << endl;

        // Not behind map? Good, then add to map.
        p->pMMData = new MapMakerData();
        mMap.vpPoints.push_back(p);

        // Construct first two measurements and insert into relevant DBs:
        Measurement mFirst;
        mFirst.nLevel = 0;
        mFirst.Source = Measurement::SRC_ROOT;
        mFirst.v2RootPos = vec(vMapPointsCandidates[addPointsNum]);
        mFirst.bSubPix = true;
        pkFirst->mMeasurements[p] = mFirst;
        p->pMMData->sMeasurementKFs.insert(pkFirst);

        Measurement mSecond;
        mSecond.nLevel = 0;
        mSecond.Source = Measurement::SRC_TRAIL;
        mSecond.v2RootPos = vec(vMapPointsCandidates[addPointsNum]);
        mSecond.bSubPix = true;
        pkSecond->mMeasurements[p] = mSecond;
        p->pMMData->sMeasurementKFs.insert(pkSecond);

    }



//    AddSomeMapPoints(0);
//    AddSomeMapPoints(3);
//    AddSomeMapPoints(1);
//    AddSomeMapPoints(2);

//    mbBundleConverged_Full = false;
//    mbBundleConverged_Recent = false;

//    while(!mbBundleConverged_Full)
//    {
//        BundleAdjustAll();
//        if(mbResetRequested)
//            return false;
//    }

    // Rotate and translate the map so the dominant plane is at z=0:
    ApplyGlobalTransformationToMap(CalcPlaneAligner());
    mMap.bGood = true;
//    se3TrackerPose = pkSecond->se3CfromW;
    se3TrackerPose = pkFirst->se3CfromW;

    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;
//    for (int i = 0; i < mMap.vpPoints.size(); i++) {
//        MapPoint *point = mMap.vpPoints[i];
//        cout << "[" << point->v3WorldPos[0] << ", " << point->v3WorldPos[1] << ", " << point->v3WorldPos[2] << "]" << endl;
//    }

    return true;
}

/*
 * 这个方法用来取代AddSomeMapPoints添加地图点
 * @param vMapPointsCandidates 存储Candidates的临时变量
 * @param nLevel 第几层金字塔
 */
void MapMaker::AddSomeMapPointsForFast(std::vector<CVD::ImageRef> &vMapPointsCandidates, int nLevel)
{
    KeyFrame &kSrc = *(mMap.vpKeyFrames[mMap.vpKeyFrames.size() - 1]); // The new keyframe
//    KeyFrame &kTarget = *(ClosestKeyFrame(kSrc));
    Level &l = kSrc.aLevels[nLevel];

//    cout << "l.vCandidates.size()111 " << l.vCandidates.size() << endl;
    ThinCandidates(kSrc, nLevel);
//    cout << "l.vCandidates.size()222 " << l.vCandidates.size() << endl;

    for(unsigned int nCandidate = 0; nCandidate < l.vCandidates.size(); nCandidate++)
    {
//        int nLevelScale = LevelScale(nLevel);
        Candidate &candidate = kSrc.aLevels[nLevel].vCandidates[nCandidate];
        ImageRef irLevelPos = candidate.irLevelPos;//在level层的像素坐标
        Vector<2> v2RootPos = LevelZeroPos(irLevelPos, nLevel);//将level层的像素坐标转换到第0层  也就是需要乘一个数
//        Vector<3> v3Ray_SC = unproject(mCamera.UnProject(v2RootPos));//将第0层的像素坐标转换到矫正后的相机坐标，并且齐次化
        vMapPointsCandidates.push_back(ir(v2RootPos));
    }
}

bool MapMaker::InitFromStereoFAST(KeyFrame &kF,
                              KeyFrame &kS,
                              vector<pair<ImageRef, ImageRef> > &vTrailMatches,
                              SE3<> &se3TrackerPose)
{
    //    if (!mMap.vpPoints.empty()) {
    //        cout << "mMap.vpPoints.size() " << mMap.vpPoints.size() << endl;
    //    }

        //cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

        cout << "  MapMakerqqq: made initial map with " << mMap.vpPoints.size() << " points." << endl;
        mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

        mCamera.SetImageSize(kF.aLevels[0].im.size());

        cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

        vector<HomographyMatch> vMatches;
        for(unsigned int i=0; i<vTrailMatches.size(); i++)
        {
            HomographyMatch m;
            m.v2CamPlaneFirst = mCamera.UnProject(vTrailMatches[i].first);
//            cout << "m.v2CamPlaneFirst: " << endl << m.v2CamPlaneFirst << endl;
            m.v2CamPlaneSecond = mCamera.UnProject(vTrailMatches[i].second);
//            cout << "m.v2CamPlaneSecond: " << endl << m.v2CamPlaneSecond << endl;
            m.m2PixelProjectionJac = mCamera.GetProjectionDerivs();
//            cout << "m.m2PixelProjectionJac: " << endl << m.m2PixelProjectionJac << endl;
            vMatches.push_back(m);
        }

        cout << "vMatches.size()0 " << vMatches.size() << endl;


        SE3<> se3;
        bool bGood;
        HomographyInit HomographyInit;
        bGood = HomographyInit.Compute2(vMatches, 5.0, se3, 0);
        cout << "se3: " << se3 << endl;

        if (isnan(se3.get_translation()[0]) ||  isnan(se3.get_translation()[1]) || isnan(se3.get_translation()[2])) {
            cout << "qqqqqqqqqqqqqqqqqqqqqq" << endl;
            return false;
        }

        if (isnan(se3.get_rotation().get_matrix()[0][0]) || isnan(se3.get_rotation().get_matrix()[0][1]) || isnan(se3.get_rotation().get_matrix()[0][2])
                || isnan(se3.get_rotation().get_matrix()[1][0]) || isnan(se3.get_rotation().get_matrix()[1][1]) || isnan(se3.get_rotation().get_matrix()[1][2])
                || isnan(se3.get_rotation().get_matrix()[2][0]) || isnan(se3.get_rotation().get_matrix()[2][1]) || isnan(se3.get_rotation().get_matrix()[2][2])) {
            cout << "wwwwwwwwwwwwwwwwwwwww" << endl;
            return false;
        }

        cout << "vMatches.size()1 " << vMatches.size() << endl;
        if(!bGood)
        {
            cout << "  Could not init from stereo pair, try again." << endl;
            return false;
        }

        // Check that the initialiser estimated a non-zero baseline
        double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
        if(dTransMagn == 0)
        {
            cout << "  Estimated zero baseline from stereo pair, try again." << endl;
            return false;
        }
        // change the scale of the map so the second camera is wiggleScale away from the first
        se3.get_translation() *= mdWiggleScale/dTransMagn;


        KeyFrame *pkFirst = new KeyFrame();
        KeyFrame *pkSecond = new KeyFrame();
        *pkFirst = kF;
        *pkSecond = kS;

        pkFirst->bFixed = true;
        pkFirst->se3CfromW = SE3<>();

        pkSecond->bFixed = false;
        pkSecond->se3CfromW = se3;

        // Construct map from the stereo matches.
        PatchFinder finder;

        cout << "vMatches.size()3 " << vMatches.size() << endl;

        bool first = true;
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            MapPoint *p = new MapPoint();

            // Patch source stuff:
            p->pPatchSourceKF = pkFirst;
            p->nSourceLevel = 0;
            p->v3Normal_NC = makeVector( 0,0,-1);
            p->irCenter = vTrailMatches[i].first;
            p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
    //        if (i == 0) {
    //            cout << mCamera.UnProject(p->irCenter)[0] << " " << mCamera.UnProject(p->irCenter)[1] << " "
    //                                                      << p->v3Center_NC[0] << " " << p->v3Center_NC[1] << " " << p->v3Center_NC[2] << endl;
    //        }
            p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
            p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
            normalize(p->v3Center_NC);
            normalize(p->v3OneDownFromCenter_NC);
            normalize(p->v3OneRightFromCenter_NC);
            p->RefreshPixelVectors();

            // Do sub-pixel alignment on the second image
            finder.MakeTemplateCoarseNoWarp(*p);
            finder.MakeSubPixTemplate();
            finder.SetSubPixPos(vec(vTrailMatches[i].second));
            bool bGood = finder.IterateSubPixToConvergence(*pkSecond,10);
            if(!bGood)
            {
                delete p; continue;
            }

            // Triangulate point:
            Vector<2> v2SecondPos = finder.GetSubPixPos();
//            if (first) {
//                cout << "v2SecondPos: " << v2SecondPos << endl;
//            }
    //        cout << vMatches[i].v2CamPlaneFirst[0] << " " << vMatches[i].v2CamPlaneFirst[1] << endl;第一帧
    //        cout << v2SecondPos[0] << " " << v2SecondPos[1] << endl;
    //        cout << mCamera.UnProject(v2SecondPos)[0] << " " << mCamera.UnProject(v2SecondPos)[1] << endl;
            p->v3WorldPos = ReprojectPoint(se3, mCamera.UnProject(v2SecondPos), vMatches[i].v2CamPlaneFirst);
            //cout << "vMatches[i].v2CamPlaneFirst1: " << endl << vMatches[i].v2CamPlaneFirst << endl;
            if(p->v3WorldPos[2] < 0.0)
            {
                //delete p; continue;
                p->v3WorldPos[2] = 0.5; continue;
            }
            //cout << p->v3WorldPos[0] << " " << p->v3WorldPos[1] << " " << p->v3WorldPos[2] << endl;
    //        for (int i = 1; i < 432; i++) {
    //            p->v3WorldPos[0] = mCamera.UnProject(v2SecondPos)[0];
    //            p->v3WorldPos[1] = mCamera.UnProject(v2SecondPos)[1];
    //            p->v3WorldPos[1] = 0.5;
    //        }

            // Not behind map? Good, then add to map.
            p->pMMData = new MapMakerData();
            mMap.vpPoints.push_back(p);

            // Construct first two measurements and insert into relevant DBs:
            Measurement mFirst;
            mFirst.nLevel = 0;
            mFirst.Source = Measurement::SRC_ROOT;
            mFirst.v2RootPos = vec(vTrailMatches[i].first);
            mFirst.bSubPix = true;
            pkFirst->mMeasurements[p] = mFirst;
            p->pMMData->sMeasurementKFs.insert(pkFirst);

            Measurement mSecond;
            mSecond.nLevel = 0;
            mSecond.Source = Measurement::SRC_TRAIL;
            mSecond.v2RootPos = finder.GetSubPixPos();
            mSecond.bSubPix = true;
            pkSecond->mMeasurements[p] = mSecond;
            p->pMMData->sMeasurementKFs.insert(pkSecond);
        }


    mMap.vpKeyFrames.push_back(pkFirst);
    mMap.vpKeyFrames.push_back(pkSecond);
    pkFirst->MakeKeyFrame_Rest();
    pkSecond->MakeKeyFrame_Rest();

    cout << "  MapMaker110: made initial map with " << mMap.vpPoints.size() << " points." << endl;
    //小于10个则重新初始化
//    if (mMap.vpPoints.size() < 10) {
//        return false;
//    }

    for(int i=0; i<5; i++) {
        if (!BundleAdjustAllFAST()) {
            return false;
        }
    }

    cout << "  MapMaker119: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    // Estimate the feature depth distribution in the first two key-frames
    // (Needed for epipolar search)
    RefreshSceneDepth(pkFirst);
    RefreshSceneDepth(pkSecond);
    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;


//    std::vector<CVD::ImageRef> vMapPointsCandidates;//存储Candidates的临时变量
//    AddSomeMapPointsForFast (vMapPointsCandidates, 0);
//    cout << "vMapPointsCandidates0: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 3);
//    cout << "vMapPointsCandidates1: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 1);
//    cout << "vMapPointsCandidates2: " << vMapPointsCandidates.size() << endl;
//    AddSomeMapPointsForFast (vMapPointsCandidates, 2);
//    cout << "vMapPointsCandidates3: " << vMapPointsCandidates.size() << endl;

    //cout << "  MapMaker0: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    AddSomeMapPoints(0);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[0].vCandidates.size() << endl;
    AddSomeMapPoints(3);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[3].vCandidates.size() << endl;
    AddSomeMapPoints(1);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[1].vCandidates.size() << endl;
    AddSomeMapPoints(2);
    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[2].vCandidates.size() << endl;
    //cout << "SUM: " << pkSecond->aLevels[0].vCandidates.size() + pkSecond->aLevels[1].vCandidates.size()
    //        + pkSecond->aLevels[2].vCandidates.size() + pkSecond->aLevels[3].vCandidates.size() << endl;



    mbBundleConverged_Full = false;
    mbBundleConverged_Recent = false;

    while(!mbBundleConverged_Full)
    {
        BundleAdjustAll();
        if(mbResetRequested)
            return false;
    }

    // Rotate and translate the map so the dominant plane is at z=0:
    ApplyGlobalTransformationToMap(CalcPlaneAligner());
    mMap.bGood = true;
    se3TrackerPose = pkSecond->se3CfromW;

    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;

    return true;
}

// Perform bundle adjustment on all keyframes, all map points
bool MapMaker::BundleAdjustAllFAST()
{
  // construct the sets of kfs/points to be adjusted:
  // in this case, all of them
  set<KeyFrame*> sAdj;
  set<KeyFrame*> sFixed;
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    if(mMap.vpKeyFrames[i]->bFixed)
      sFixed.insert(mMap.vpKeyFrames[i]);
    else
      sAdj.insert(mMap.vpKeyFrames[i]);

  set<MapPoint*> sMapPoints;
  for(unsigned int i=0; i<mMap.vpPoints.size();i++)
    sMapPoints.insert(mMap.vpPoints[i]);

  return BundleAdjustFAST(sAdj, sFixed, sMapPoints, false);
}

// Common bundle adjustment code. This creates a bundle-adjust instance, populates it, and runs it.
bool MapMaker::BundleAdjustFAST(set<KeyFrame*> sAdjustSet, set<KeyFrame*> sFixedSet, set<MapPoint*> sMapPoints, bool bRecent)
{
  Bundle b(mCamera);   // Our bundle adjuster
  mbBundleRunning = true;
  mbBundleRunningIsRecent = bRecent;

  // The bundle adjuster does different accounting of keyframes and map points;
  // Translation maps are stored:
  map<MapPoint*, int> mPoint_BundleID;
  map<int, MapPoint*> mBundleID_Point;
  map<KeyFrame*, int> mView_BundleID;
  map<int, KeyFrame*> mBundleID_View;

  // Add the keyframes' poses to the bundle adjuster. Two parts: first nonfixed, then fixed.
  for(set<KeyFrame*>::iterator it = sAdjustSet.begin(); it!= sAdjustSet.end(); it++)
    {
      int nBundleID = b.AddCamera((*it)->se3CfromW, (*it)->bFixed);
      mView_BundleID[*it] = nBundleID;
      mBundleID_View[nBundleID] = *it;
    }
  for(set<KeyFrame*>::iterator it = sFixedSet.begin(); it!= sFixedSet.end(); it++)
    {
      int nBundleID = b.AddCamera((*it)->se3CfromW, true);
      mView_BundleID[*it] = nBundleID;
      mBundleID_View[nBundleID] = *it;
    }

  // Add the points' 3D position
  for(set<MapPoint*>::iterator it = sMapPoints.begin(); it!=sMapPoints.end(); it++)
    {
      int nBundleID = b.AddPointFAST((*it)->v3WorldPos);
      if (nBundleID == -1) {
          return false;
      }
      mPoint_BundleID[*it] = nBundleID;
      mBundleID_Point[nBundleID] = *it;
    }

  // Add the relevant point-in-keyframe measurements
  for(unsigned int i=0; i<mMap.vpKeyFrames.size(); i++)
    {
      if(mView_BundleID.count(mMap.vpKeyFrames[i]) == 0)
    continue;

      int nKF_BundleID = mView_BundleID[mMap.vpKeyFrames[i]];
      for(meas_it it= mMap.vpKeyFrames[i]->mMeasurements.begin();
      it!= mMap.vpKeyFrames[i]->mMeasurements.end();
      it++)
    {
      if(mPoint_BundleID.count(it->first) == 0)
        continue;
      int nPoint_BundleID = mPoint_BundleID[it->first];
      b.AddMeas(nKF_BundleID, nPoint_BundleID, it->second.v2RootPos, LevelScale(it->second.nLevel) * LevelScale(it->second.nLevel));
    }
    }

  // Run the bundle adjuster. This returns the number of successful iterations
  int nAccepted = b.Compute(&mbBundleAbortRequested);

  if(nAccepted < 0)
    {
      // Crap: - LM Ran into a serious problem!
      // This is probably because the initial stereo was messed up.
      // Get rid of this map and start again!
      cout << "!! MapMaker: Cholesky failure in bundle adjust. " << endl
       << "   The map is probably corrupt: Ditching the map. " << endl;
      mbResetRequested = true;
      return false;
    }

  // Bundle adjustment did some updates, apply these to the map
  if(nAccepted > 0)
    {

      for(map<MapPoint*,int>::iterator itr = mPoint_BundleID.begin();
      itr!=mPoint_BundleID.end();
      itr++)
    itr->first->v3WorldPos = b.GetPoint(itr->second);

      for(map<KeyFrame*,int>::iterator itr = mView_BundleID.begin();
      itr!=mView_BundleID.end();
      itr++)
    itr->first->se3CfromW = b.GetCamera(itr->second);
      if(bRecent)
    mbBundleConverged_Recent = false;
      mbBundleConverged_Full = false;
    };

  if(b.Converged())
    {
      mbBundleConverged_Recent = true;
      if(!bRecent)
    mbBundleConverged_Full = true;
    }

  mbBundleRunning = false;
  mbBundleAbortRequested = false;

  // Handle outlier measurements:
  vector<pair<int,int> > vOutliers_PC_pair = b.GetOutlierMeasurements();
  for(unsigned int i=0; i<vOutliers_PC_pair.size(); i++)
    {
      MapPoint *pp = mBundleID_Point[vOutliers_PC_pair[i].first];
      KeyFrame *pk = mBundleID_View[vOutliers_PC_pair[i].second];
      Measurement &m = pk->mMeasurements[pp];
      if(pp->pMMData->GoodMeasCount() <= 2 || m.Source == Measurement::SRC_ROOT)   // Is the original source kf considered an outlier? That's bad.
    pp->bBad = true;
      else
    {
      // Do we retry it? Depends where it came from!!
      if(m.Source == Measurement::SRC_TRACKER || m.Source == Measurement::SRC_EPIPOLAR)
        mvFailureQueue.push_back(pair<KeyFrame*,MapPoint*>(pk,pp));
      else
        pp->pMMData->sNeverRetryKFs.insert(pk);
      pk->mMeasurements.erase(pp);
      pp->pMMData->sMeasurementKFs.erase(pk);
    }
    }
  return true;
}

//bool MapMaker::InitFromStereoFAST2(KeyFrame kF,
//                              KeyFrame kS,
//                              vector<pair<ImageRef, ImageRef> > vTrailMatches,
//                              SE3<> se3TrackerPose)
//{
//    //    if (!mMap.vpPoints.empty()) {
//    //        cout << "mMap.vpPoints.size() " << mMap.vpPoints.size() << endl;
//    //    }

//        //cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

//        cout << "  MapMakerqqq: made initial map with " << mMap.vpPoints.size() << " points." << endl;
//        mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

//        mCamera.SetImageSize(kF.aLevels[0].im.size());

//        cout << "vTrailMatches.size() " << vTrailMatches.size() << endl;

//        vector<HomographyMatch> vMatches;
//        for(unsigned int i=0; i<vTrailMatches.size(); i++)
//        {
//            HomographyMatch m;
//            m.v2CamPlaneFirst = mCamera.UnProject(vTrailMatches[i].first);
//            m.v2CamPlaneSecond = mCamera.UnProject(vTrailMatches[i].second);
//            m.m2PixelProjectionJac = mCamera.GetProjectionDerivs();
//            vMatches.push_back(m);
//        }

//        cout << "vMatches.size()0 " << vMatches.size() << endl;


//        SE3<> se3;
//        bool bGood;
//        HomographyInit HomographyInit;
//        bGood = HomographyInit.Compute(vMatches, 5.0, se3);
//        cout << "vMatches.size()1 " << vMatches.size() << endl;
//        if(!bGood)
//        {
//            cout << "  Could not init from stereo pair, try again." << endl;
//            return false;
//        }

//        // Check that the initialiser estimated a non-zero baseline
//        double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
//        if(dTransMagn == 0)
//        {
//            cout << "  Estimated zero baseline from stereo pair, try again." << endl;
//            return false;
//        }
//        // change the scale of the map so the second camera is wiggleScale away from the first
//        se3.get_translation() *= mdWiggleScale/dTransMagn;


//        KeyFrame pkFirst;
//        KeyFrame pkSecond;
//        pkFirst = kF;
//        pkSecond = kS;

//        pkFirst.bFixed = true;
//        pkFirst.se3CfromW = SE3<>();

//        pkSecond.bFixed = false;
//        pkSecond.se3CfromW = se3;

//        // Construct map from the stereo matches.
//        PatchFinder finder;

//        cout << "vMatches.size()3 " << vMatches.size() << endl;

//        for(unsigned int i=0; i<vMatches.size(); i++)
//        {
//            MapPoint *p = new MapPoint();

//            // Patch source stuff:
//            p->pPatchSourceKF = pkFirst;
//            p->nSourceLevel = 0;
//            p->v3Normal_NC = makeVector( 0,0,-1);
//            p->irCenter = vTrailMatches[i].first;
//            p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
//    //        if (i == 0) {
//    //            cout << mCamera.UnProject(p->irCenter)[0] << " " << mCamera.UnProject(p->irCenter)[1] << " "
//    //                                                      << p->v3Center_NC[0] << " " << p->v3Center_NC[1] << " " << p->v3Center_NC[2] << endl;
//    //        }
//            p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
//            p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
//            normalize(p->v3Center_NC);
//            normalize(p->v3OneDownFromCenter_NC);
//            normalize(p->v3OneRightFromCenter_NC);
//            p->RefreshPixelVectors();

//            // Do sub-pixel alignment on the second image
//            finder.MakeTemplateCoarseNoWarp(*p);
//            finder.MakeSubPixTemplate();
//            finder.SetSubPixPos(vec(vTrailMatches[i].second));
//            bool bGood = finder.IterateSubPixToConvergence(*pkSecond,10);
//            if(!bGood)
//            {
//                delete p; continue;
//            }

//            // Triangulate point:
//            Vector<2> v2SecondPos = finder.GetSubPixPos();
//    //        cout << vMatches[i].v2CamPlaneFirst[0] << " " << vMatches[i].v2CamPlaneFirst[1] << endl;第一帧
//    //        cout << v2SecondPos[0] << " " << v2SecondPos[1] << endl;
//    //        cout << mCamera.UnProject(v2SecondPos)[0] << " " << mCamera.UnProject(v2SecondPos)[1] << endl;
//            p->v3WorldPos = ReprojectPoint(se3, mCamera.UnProject(v2SecondPos), vMatches[i].v2CamPlaneFirst);
//            if(p->v3WorldPos[2] < 0.0)
//            {
//                delete p; continue;
//            }
//            //cout << p->v3WorldPos[0] << " " << p->v3WorldPos[1] << " " << p->v3WorldPos[2] << endl;
//    //        for (int i = 1; i < 432; i++) {
//    //            p->v3WorldPos[0] = mCamera.UnProject(v2SecondPos)[0];
//    //            p->v3WorldPos[1] = mCamera.UnProject(v2SecondPos)[1];
//    //            p->v3WorldPos[1] = 0.5;
//    //        }

//            // Not behind map? Good, then add to map.
//            p->pMMData = new MapMakerData();
//            mMap.vpPoints.push_back(p);

//            // Construct first two measurements and insert into relevant DBs:
//            Measurement mFirst;
//            mFirst.nLevel = 0;
//            mFirst.Source = Measurement::SRC_ROOT;
//            mFirst.v2RootPos = vec(vTrailMatches[i].first);
//            mFirst.bSubPix = true;
//            pkFirst->mMeasurements[p] = mFirst;
//            p->pMMData->sMeasurementKFs.insert(pkFirst);

//            Measurement mSecond;
//            mSecond.nLevel = 0;
//            mSecond.Source = Measurement::SRC_TRAIL;
//            mSecond.v2RootPos = finder.GetSubPixPos();
//            mSecond.bSubPix = true;
//            pkSecond->mMeasurements[p] = mSecond;
//            p->pMMData->sMeasurementKFs.insert(pkSecond);
//        }


//    mMap.vpKeyFrames.push_back(pkFirst);
//    mMap.vpKeyFrames.push_back(pkSecond);
//    pkFirst->MakeKeyFrame_Rest();
//    pkSecond->MakeKeyFrame_Rest();

//    cout << "  MapMaker110: made initial map with " << mMap.vpPoints.size() << " points." << endl;

//    for(int i=0; i<5; i++) {
//        if (!BundleAdjustAllFAST()) {
//            return false;
//        }
//    }

//    cout << "  MapMaker119: made initial map with " << mMap.vpPoints.size() << " points." << endl;

//    // Estimate the feature depth distribution in the first two key-frames
//    // (Needed for epipolar search)
//    RefreshSceneDepth(pkFirst);
//    RefreshSceneDepth(pkSecond);
//    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;


////    std::vector<CVD::ImageRef> vMapPointsCandidates;//存储Candidates的临时变量
////    AddSomeMapPointsForFast (vMapPointsCandidates, 0);
////    cout << "vMapPointsCandidates0: " << vMapPointsCandidates.size() << endl;
////    AddSomeMapPointsForFast (vMapPointsCandidates, 3);
////    cout << "vMapPointsCandidates1: " << vMapPointsCandidates.size() << endl;
////    AddSomeMapPointsForFast (vMapPointsCandidates, 1);
////    cout << "vMapPointsCandidates2: " << vMapPointsCandidates.size() << endl;
////    AddSomeMapPointsForFast (vMapPointsCandidates, 2);
////    cout << "vMapPointsCandidates3: " << vMapPointsCandidates.size() << endl;

//    //cout << "  MapMaker0: made initial map with " << mMap.vpPoints.size() << " points." << endl;

//    AddSomeMapPoints(0);
//    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[0].vCandidates.size() << endl;
//    AddSomeMapPoints(3);
//    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[3].vCandidates.size() << endl;
//    AddSomeMapPoints(1);
//    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[1].vCandidates.size() << endl;
//    AddSomeMapPoints(2);
//    //cout << "pkSecond->aLevels[nLevel].vCandidates.size()  " << pkSecond->aLevels[2].vCandidates.size() << endl;
//    //cout << "SUM: " << pkSecond->aLevels[0].vCandidates.size() + pkSecond->aLevels[1].vCandidates.size()
//    //        + pkSecond->aLevels[2].vCandidates.size() + pkSecond->aLevels[3].vCandidates.size() << endl;



//    mbBundleConverged_Full = false;
//    mbBundleConverged_Recent = false;

//    while(!mbBundleConverged_Full)
//    {
//        BundleAdjustAll();
//        if(mbResetRequested)
//            return false;
//    }

//    // Rotate and translate the map so the dominant plane is at z=0:
//    ApplyGlobalTransformationToMap(CalcPlaneAligner());
//    mMap.bGood = true;
//    se3TrackerPose = pkSecond->se3CfromW;

//    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;

//    return true;
//}


// 前后两帧备份副本
// //根据两帧，但是替换深度
//bool MapMaker::InitFromStereo2(cv::Mat src, KeyFrame &kF,
//                              KeyFrame &kS,
//                              vector<pair<ImageRef, ImageRef> > &vTrailMatches,
//                              SE3<> &se3TrackerPose)
//{


//    //compute vanishing point
//    mvVanishingPoints.clear();
//    int numVps = 0;
//    DetectVanishingPoints(src, mvVanishingPoints);
//    numVps = mvVanishingPoints.size();

//    cout<<"numVps: "<<numVps<<endl;





//    mdWiggleScale = *mgvdWiggleScale; // Cache this for the new map.

//    mCamera.SetImageSize(kF.aLevels[0].im.size());


//    vector<HomographyMatch> vMatches;
//    for(unsigned int i=0; i<vTrailMatches.size(); i++)
//    {
//        HomographyMatch m;
//        m.v2CamPlaneFirst = mCamera.UnProject(vTrailMatches[i].first);
//        m.v2CamPlaneSecond = mCamera.UnProject(vTrailMatches[i].second);
//        m.m2PixelProjectionJac = mCamera.GetProjectionDerivs();
//        vMatches.push_back(m);
//    }

//    SE3<> se3;
//    bool bGood;
//    HomographyInit HomographyInit;
//    bGood = HomographyInit.Compute(vMatches, 5.0, se3);
//    if(!bGood)
//    {
//        cout << "  Could not init from stereo pair, try again." << endl;
//        return false;
//    }

//    // Check that the initialiser estimated a non-zero baseline
//    double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
//    if(dTransMagn == 0)
//    {
//        cout << "  Estimated zero baseline from stereo pair, try again." << endl;
//        return false;
//    }
//    // change the scale of the map so the second camera is wiggleScale away from the first
//    se3.get_translation() *= mdWiggleScale/dTransMagn;


//    KeyFrame *pkFirst = new KeyFrame();
//    KeyFrame *pkSecond = new KeyFrame();
//    *pkFirst = kF;
//    *pkSecond = kS;

//    pkFirst->bFixed = true;
//    pkFirst->se3CfromW = SE3<>();

//    pkSecond->bFixed = false;
//    pkSecond->se3CfromW = se3;

//    // Construct map from the stereo matches.
//    PatchFinder finder;

//    for (unsigned int i=0; i<vMatches.size(); i++) {

//    }
//    double maxDist=0, minDist=100000;
//    if (numVps>0) {
//        hasNewDetectedVPs = true;
//        //对每一个特征点去计算该特征点到灭点的距离之和
//        for (unsigned int i=0; i<vMatches.size(); i++)
//        {
//            double dist = 0;

//            for (int vi=0; vi<numVps; vi++)
//            {
//                dist += sqrt((vTrailMatches[i].first.x-mvVanishingPoints[vi].at<float>(0,0)) * (vTrailMatches[i].first.x-mvVanishingPoints[vi].at<float>(0,0)) + (vTrailMatches[i].first.y-mvVanishingPoints[vi].at<float>(0,1)) * (vTrailMatches[i].first.y-mvVanishingPoints[vi].at<float>(0,1)));
//            }
//            if (dist>maxDist){
//                maxDist = dist;
//            }
//            if (dist<minDist) {
//                minDist = dist;
//            }
//        }
//        //cout << minDist << "   " << maxDist << endl;
//    }

//    for(unsigned int i=0; i<vMatches.size(); i++)
//    {
//        MapPoint *p = new MapPoint();

//        // Patch source stuff:
//        p->pPatchSourceKF = pkFirst;
//        p->nSourceLevel = 0;
//        p->v3Normal_NC = makeVector( 0,0,-1);
//        p->irCenter = vTrailMatches[i].first;
//        p->v3Center_NC = unproject(mCamera.UnProject(p->irCenter));
//        p->v3OneDownFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(0,1)));
//        p->v3OneRightFromCenter_NC = unproject(mCamera.UnProject(p->irCenter + ImageRef(1,0)));
//        normalize(p->v3Center_NC);
//        normalize(p->v3OneDownFromCenter_NC);
//        normalize(p->v3OneRightFromCenter_NC);
//        p->RefreshPixelVectors();

//        // Do sub-pixel alignment on the second image
//        finder.MakeTemplateCoarseNoWarp(*p);
//        finder.MakeSubPixTemplate();
//        finder.SetSubPixPos(vec(vTrailMatches[i].second));
//        bool bGood = finder.IterateSubPixToConvergence(*pkSecond,10);
//        if(!bGood)
//        {
//            delete p; continue;
//        }

//        // Triangulate point:
//        //Vector<2> v2SecondPos = finder.GetSubPixPos();
////        p->v3WorldPos = ReprojectPoint(se3, mCamera.UnProject(v2SecondPos), vMatches[i].v2CamPlaneFirst);
////        if(p->v3WorldPos[2] < 0.0)
////        {
////            delete p; continue;
////        }


//        random_device rd;
//        mt19937 gen(rd());
//        normal_distribution<double> normal(1,0.125);
//        double z_depth;
//        if (numVps==0) {
//            z_depth = normal(gen);
//        }
//        else {
//            double dist = 0;
//            for (int vi=0; vi<numVps; vi++){
//                dist += sqrt((vTrailMatches[i].first.x-mvVanishingPoints[vi].at<float>(0,0)) * (vTrailMatches[i].first.x-mvVanishingPoints[vi].at<float>(0,0)) + (vTrailMatches[i].first.y-mvVanishingPoints[vi].at<float>(0,1)) * (vTrailMatches[i].first.y-mvVanishingPoints[vi].at<float>(0,1)));
//            }
//            z_depth = (maxDist-dist)/(maxDist-minDist)+0.5;//+normal(gen)/10;
//        }

//        p->v3WorldPos[0] = vMatches[i].v2CamPlaneFirst[0];
//        p->v3WorldPos[1] = vMatches[i].v2CamPlaneFirst[0];
//        p->v3WorldPos[2] = z_depth;

//        // Not behind map? Good, then add to map.
//        p->pMMData = new MapMakerData();
//        mMap.vpPoints.push_back(p);

//        // Construct first two measurements and insert into relevant DBs:
//        Measurement mFirst;
//        mFirst.nLevel = 0;
//        mFirst.Source = Measurement::SRC_ROOT;
//        mFirst.v2RootPos = vec(vTrailMatches[i].first);
//        mFirst.bSubPix = true;
//        pkFirst->mMeasurements[p] = mFirst;
//        p->pMMData->sMeasurementKFs.insert(pkFirst);

//        Measurement mSecond;
//        mSecond.nLevel = 0;
//        mSecond.Source = Measurement::SRC_TRAIL;
//        mSecond.v2RootPos = finder.GetSubPixPos();
//        mSecond.bSubPix = true;
//        pkSecond->mMeasurements[p] = mSecond;
//        p->pMMData->sMeasurementKFs.insert(pkSecond);
//    }

//    mMap.vpKeyFrames.push_back(pkFirst);
//    mMap.vpKeyFrames.push_back(pkSecond);
//    pkFirst->MakeKeyFrame_Rest();
//    pkSecond->MakeKeyFrame_Rest();

//    for(int i=0; i<5; i++)
//        BundleAdjustAll();

//    // Estimate the feature depth distribution in the first two key-frames
//    // (Needed for epipolar search)
//    RefreshSceneDepth(pkFirst);
//    RefreshSceneDepth(pkSecond);
//    mdWiggleScaleDepthNormalized = mdWiggleScale / pkFirst->dSceneDepthMean;


//    AddSomeMapPoints(0);
//    AddSomeMapPoints(3);
//    AddSomeMapPoints(1);
//    AddSomeMapPoints(2);

//    mbBundleConverged_Full = false;
//    mbBundleConverged_Recent = false;

//    while(!mbBundleConverged_Full)
//    {
//        BundleAdjustAll();
//        if(mbResetRequested)
//            return false;
//    }

//    // Rotate and translate the map so the dominant plane is at z=0:
//    ApplyGlobalTransformationToMap(CalcPlaneAligner());
//    mMap.bGood = true;
//    se3TrackerPose = pkSecond->se3CfromW;

//    cout << "  MapMaker: made initial map with " << mMap.vpPoints.size() << " points." << endl;
////    for (int i = 0; i < mMap.vpPoints.size(); i++) {
////        MapPoint *point = mMap.vpPoints[i];
////        cout << "[" << point->v3WorldPos[0] << ", " << point->v3WorldPos[1] << ", " << point->v3WorldPos[2] << "]" << endl;
////    }

//    return true;
//}
