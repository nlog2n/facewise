using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.VideoSurveillance;
using Emgu.Util;


namespace FaceSmart
{
    public class MotionDetector
    {
        //camera capture
        //Capture _capture;
        //Image<Bgr, Byte> currentFrame;

        MotionHistory _motionHistory;
        IBGFGDetector<Bgr> _forgroundDetector = null;

        public MotionDetector()
        {
            //Initialize the capture device
            //_capture = new Capture();
            //_capture.QueryFrame();

            _motionHistory = new MotionHistory(
                     1, //in second, the duration of motion history you wants to keep
                     0.05, //in second, maxDelta for cvCalcMotionGradient
                     0.5); //in second, minDelta for cvCalcMotionGradient
        }


        // return:  the amount of motions found on the current image
        // note: may also touch the image
        public int DetectMotion(ref Image<Bgr, Byte> image)
        {
            int result = 0;

            //image = image.Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            //image = image.PyrUp(); //reduce noise from the image

            using (MemStorage storage = new MemStorage()) //create storage for motion components
            {
                if (_forgroundDetector == null)
                {
                    //_forgroundDetector = new BGCodeBookModel<Bgr>();
                    //_forgroundDetector = new FGDetector<Bgr>(Emgu.CV.CvEnum.FORGROUND_DETECTOR_TYPE.FGD);
                    _forgroundDetector = new BGStatModel<Bgr>(image, Emgu.CV.CvEnum.BG_STAT_TYPE.GAUSSIAN_BG_MODEL);
                }

                _forgroundDetector.Update(image);

                //update the motion history
                //_motionHistory.Update(image.Convert<Gray, Byte>());
                _motionHistory.Update(_forgroundDetector.ForgroundMask);

                #region get a copy of the motion mask and enhance its color
                double[] minValues, maxValues;
                Point[] minLoc, maxLoc;
                _motionHistory.Mask.MinMax(out minValues, out maxValues, out minLoc, out maxLoc);
                Image<Gray, Byte> motionMask = _motionHistory.Mask.Mul(255.0 / maxValues[0]);
                #endregion

                //create the motion image 
                Image<Bgr, Byte> motionImage = new Image<Bgr, byte>(motionMask.Size);
                //display the motion pixels in blue (first channel)
                motionImage[0] = motionMask;

                //Threshold to define a motion area, reduce the value to detect smaller motion
                // fanghui: only detect face size and palm size
                double minArea = 100; 

                storage.Clear(); //clear the storage
                Seq<MCvConnectedComp> motionComponents = _motionHistory.GetMotionComponents(storage);

                //iterate through each of the motion component
                foreach (MCvConnectedComp comp in motionComponents)
                {
                    //reject the components that have small area;
                    if (comp.area < minArea) continue;

                    // find the angle and motion pixel count of the specific area
                    double angle, motionPixelCount;
                    _motionHistory.MotionInfo(comp.rect, out angle, out motionPixelCount);

                    //reject the area that contains too few motion
                    if (motionPixelCount < comp.area * 0.05) continue;

                    result++;

                    //Draw each individual motion in red
                    //DrawMotion(motionImage, comp.rect, angle, new Bgr(Color.Red));
                    DrawMotion(image, comp.rect, angle, new Bgr(Color.Red));
                }

                #region //find and draw the overall motion angle
                // fanghui: do not draw overall picture
                /*
                double overallAngle, overallMotionPixelCount;
                _motionHistory.MotionInfo(motionMask.ROI, out overallAngle, out overallMotionPixelCount);
                DrawMotion(motionImage, motionMask.ROI, overallAngle, new Bgr(Color.Green));
               */
                #endregion

                //image = motionImage;

                return (result - 1 > 0 ? (result - 1) : 0);
            }
        }


        private static void DrawMotion(Image<Bgr, Byte> image, Rectangle motionRegion, double angle, Bgr color)
        {
            float circleRadius = (motionRegion.Width + motionRegion.Height) >> 2;
            Point center = new Point(motionRegion.X + motionRegion.Width >> 1, motionRegion.Y + motionRegion.Height >> 1);

            CircleF circle = new CircleF(
               center,
               circleRadius);

            int xDirection = (int)(Math.Cos(angle * (Math.PI / 180.0)) * circleRadius);
            int yDirection = (int)(Math.Sin(angle * (Math.PI / 180.0)) * circleRadius);
            Point pointOnCircle = new Point(
                center.X + xDirection,
                center.Y - yDirection);
            LineSegment2D line = new LineSegment2D(center, pointOnCircle);

            //image.Draw(circle, color, 1);
            //image.Draw(line, color, 2);
            image.Draw(motionRegion, color, 1);
        }




    }
}