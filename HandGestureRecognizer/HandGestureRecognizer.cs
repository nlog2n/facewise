using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;

using Emgu.CV.Structure;
using Emgu.CV;
using HandGestureRecognition.SkinDetector;

namespace HandGestureRecognition
{
    public class HandGestureRecognizer 
    {
        IColorSkinDetector skinDetector;

        Image<Bgr, Byte> currentFrame;
        Image<Bgr, Byte> currentFrameCopy;
                
        Capture grabber;
        AdaptiveSkinDetector detector;
        
        int frameWidth;
        int frameHeight;
        
        Hsv hsv_min;
        Hsv hsv_max;
        Ycc YCrCb_min;
        Ycc YCrCb_max;
        
        Seq<Point> hull;
        Seq<Point> filteredHull;
        Seq<MCvConvexityDefect> defects;
        MCvConvexityDefect[] defectArray;
        Rectangle handRect;
        MCvBox2D box;
        Ellipse ellip;


        public HandGestureRecognizer()
        {
            //currentFrame = frame;
            //frameWidth = frame.Width;
            //frameHeight = frame.Height;

            detector = new AdaptiveSkinDetector(1, AdaptiveSkinDetector.MorphingMethod.NONE);
            hsv_min = new Hsv(0, 45, 0);
            hsv_max = new Hsv(20, 255, 255);
            YCrCb_min = new Ycc(0, 131, 80);
            YCrCb_max = new Ycc(255, 185, 135);
            box = new MCvBox2D();
            ellip = new Ellipse();

            //Application.Idle += new EventHandler(FrameGrabber);                        
        }

        public HandGestureRecognizer(string movfile)
        {
            // read from movie file
            //grabber = new Emgu.CV.Capture(@".\sample.MPG");
            grabber = new Emgu.CV.Capture(movfile);
            currentFrame = grabber.QueryFrame();
            frameWidth = grabber.Width;
            frameHeight = grabber.Height;            

            detector = new AdaptiveSkinDetector(1, AdaptiveSkinDetector.MorphingMethod.NONE);
            hsv_min = new Hsv(0, 45, 0); 
            hsv_max = new Hsv(20, 255, 255);            
            YCrCb_min = new Ycc(0, 131, 80);
            YCrCb_max = new Ycc(255, 185, 135);
            box = new MCvBox2D();
            ellip = new Ellipse();

            //Application.Idle += new EventHandler(FrameGrabber);                        
        }

        public bool Recognize(Image<Bgr, Byte> frame)
        {
            if (frame == null)
            {
                currentFrame = grabber.QueryFrame();
                frameWidth = grabber.Width;
                frameHeight = grabber.Height;
            }
            else
            {
                currentFrame = frame;
                frameWidth = frame.Width;
                frameHeight = frame.Height;
            }

            if (currentFrame == null) return false;

            currentFrameCopy = currentFrame.Copy();

            // Uncomment if using opencv adaptive skin detector
            //Image<Gray,Byte> skin = new Image<Gray,byte>(currentFrameCopy.Width,currentFrameCopy.Height);                
            //detector.Process(currentFrameCopy, skin);                

            // my skin detector
            skinDetector = new YCrCbSkinDetector();
            Image<Gray, Byte> skin = skinDetector.DetectSkin(currentFrameCopy, YCrCb_min, YCrCb_max);

            return ExtractContourAndHull(skin);

            //imageBoxSkin.Image = skin;  // the right image box for skin
            //imageBoxFrameGrabber.Image = currentFrame;  // the left image box for original overlay
        }
               
        private bool ExtractContourAndHull(Image<Gray, byte> skin)
        {
            using (MemStorage storage = new MemStorage())
            {
                Contour<Point> contours = skin.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
                Contour<Point> biggestContour = null;  // as face
                Contour<Point> secondbiggestContour = null;  // as hand
                Double curArea = 0;
                Double largestArea = 0;
                Double secondlargtestArea = 0;

                while (contours != null)
                {
                    curArea = contours.Area;
                    if (curArea > largestArea)  // choose smaller one because face is present
                    {
                        secondlargtestArea = largestArea;
                        largestArea = curArea;

                        secondbiggestContour = biggestContour;
                        biggestContour = contours;
                    }
                    else if (curArea > secondlargtestArea)
                    {
                        secondlargtestArea = curArea;
                        secondbiggestContour = contours;
                    }

                    contours = contours.HNext;
                }

                if (biggestContour == null || secondbiggestContour == null)
                    return false;

                // choose second largest contour as hand
                Contour<Point> currentContour = secondbiggestContour.ApproxPoly(secondbiggestContour.Perimeter * 0.0025, storage);
                currentFrame.Draw(currentContour, new Bgr(Color.LimeGreen), 2);

                // extract hand hull and defects
                hull = currentContour.GetConvexHull(Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                box = currentContour.GetMinAreaRect();
                PointF[] points = box.GetVertices();
                //handRect = box.MinAreaRect();
                //currentFrame.Draw(handRect, new Bgr(200, 0, 0), 1);

                Point[] ps = new Point[points.Length];
                for (int i = 0; i < points.Length; i++)
                    ps[i] = new Point((int)points[i].X, (int)points[i].Y);

                //currentFrame.DrawPolyline(hull.ToArray(), true, new Bgr(200, 125, 75), 2);
                //currentFrame.Draw(new CircleF(new PointF(box.center.X, box.center.Y), 3), new Bgr(200, 125, 75), 2);

                //ellip.MCvBox2D= CvInvoke.cvFitEllipse2(biggestContour.Ptr);
                //currentFrame.Draw(new Ellipse(ellip.MCvBox2D), new Bgr(Color.LavenderBlush), 3);

                PointF center;
                float radius;
                //CvInvoke.cvMinEnclosingCircle(biggestContour.Ptr, out  center, out  radius);
                //currentFrame.Draw(new CircleF(center, radius), new Bgr(Color.Gold), 2);

                //currentFrame.Draw(new CircleF(new PointF(ellip.MCvBox2D.center.X, ellip.MCvBox2D.center.Y), 3), new Bgr(100, 25, 55), 2);
                //currentFrame.Draw(ellip, new Bgr(Color.DeepPink), 2);

                //CvInvoke.cvEllipse(currentFrame, new Point((int)ellip.MCvBox2D.center.X, (int)ellip.MCvBox2D.center.Y), new System.Drawing.Size((int)ellip.MCvBox2D.size.Width, (int)ellip.MCvBox2D.size.Height), ellip.MCvBox2D.angle, 0, 360, new MCvScalar(120, 233, 88), 1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, 0);
                //currentFrame.Draw(new Ellipse(new PointF(box.center.X, box.center.Y), new SizeF(box.size.Height, box.size.Width), box.angle), new Bgr(0, 0, 0), 2);


                filteredHull = new Seq<Point>(storage);
                for (int i = 0; i < hull.Total; i++)
                {
                    if (Math.Sqrt(Math.Pow(hull[i].X - hull[i + 1].X, 2) + Math.Pow(hull[i].Y - hull[i + 1].Y, 2)) > box.size.Width / 10)
                    {
                        filteredHull.Push(hull[i]);
                    }
                }

                defects = currentContour.GetConvexityDefacts(storage, Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                defectArray = defects.ToArray();

                DrawAndComputeFingersNum();
                return true;
            }
        }


        // further identify fingers and count them (1-5)
        private int DrawAndComputeFingersNum()
        {
            int fingerNum = 0;

            #region hull drawing
            //for (int i = 0; i < filteredHull.Total; i++)
            //{
            //    PointF hullPoint = new PointF((float)filteredHull[i].X,
            //                                  (float)filteredHull[i].Y);
            //    CircleF hullCircle = new CircleF(hullPoint, 4);
            //    currentFrame.Draw(hullCircle, new Bgr(Color.Aquamarine), 2);
            //}
            #endregion

            #region defects drawing
            if (defects == null) return 0;
            for (int i = 0; i < defects.Total; i++)
            {
                PointF startPoint = new PointF((float)defectArray[i].StartPoint.X,
                                                (float)defectArray[i].StartPoint.Y);

                PointF depthPoint = new PointF((float)defectArray[i].DepthPoint.X,
                                                (float)defectArray[i].DepthPoint.Y);

                PointF endPoint = new PointF((float)defectArray[i].EndPoint.X,
                                                (float)defectArray[i].EndPoint.Y);

                LineSegment2D startDepthLine = new LineSegment2D(defectArray[i].StartPoint, defectArray[i].DepthPoint);

                LineSegment2D depthEndLine = new LineSegment2D(defectArray[i].DepthPoint, defectArray[i].EndPoint);

                CircleF startCircle = new CircleF(startPoint, 5f);

                CircleF depthCircle = new CircleF(depthPoint, 5f);

                CircleF endCircle = new CircleF(endPoint, 5f);

                //Custom heuristic based on some experiment, double check it before use
                if ((startCircle.Center.Y < box.center.Y || depthCircle.Center.Y < box.center.Y) 
                    && (startCircle.Center.Y < depthCircle.Center.Y) 
                    && (Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2)) > box.size.Height / 6.5))
                {
                    fingerNum++;
                    currentFrame.Draw(startDepthLine, new Bgr(Color.Green), 2);
                    //currentFrame.Draw(depthEndLine, new Bgr(Color.Magenta), 2);
                }

                                
                //currentFrame.Draw(startCircle, new Bgr(Color.Red), 2);
                //currentFrame.Draw(depthCircle, new Bgr(Color.Yellow), 5);
                //currentFrame.Draw(endCircle, new Bgr(Color.DarkBlue), 4);
            }
            #endregion

            #region show the number of fingers
            //MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_DUPLEX, 5d, 5d);
            //currentFrame.Draw(fingerNum.ToString(), ref font, new Point(50, 150), new Bgr(Color.White));
            #endregion

            return fingerNum;
        }
                                      
    }
}