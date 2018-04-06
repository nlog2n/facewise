// Online face detection and recognition
// Using EmguCV cross platform .Net wrapper to the Intel OpenCV image processing library for C#.Net

using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace FaceSmart
{
    public class FaceRecognizer
    {
        //Declararation of all variables, vectors and haarcascades
        //Image<Bgr, Byte> currentFrame = null;
        Capture _capture = null;

        //Load haarcascades for face detection
        HaarCascade _haarFace = new HaarCascade("haarcascade_frontalface_default.xml");
        //HaarCascade _haarEye  = new HaarCascade("haarcascade_eye.xml");


        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        List<string> labels = new List<string>();

        string path2db = ".";


        public FaceRecognizer()
        {
            //Initialize the capture device
            //_capture = new Capture();
            //_capture.QueryFrame();
        }

        public bool LoadFace(string path)
        {
            try
            {
                path2db = path;

                //Load of previus trainned faces and labels for each image
                string Labelsinfo = File.ReadAllText(path + "/TrainedFaces/TrainedLabels.txt");
                string[] Labels = Labelsinfo.Split('%');
                int NumLabels = Convert.ToInt16(Labels[0]);

                for (int tf = 1; tf < NumLabels + 1; tf++)
                {
                    string LoadFaces = "face" + tf + ".bmp";
                    trainingImages.Add(new Image<Gray, byte>(path + "/TrainedFaces/" + LoadFaces));
                    labels.Add(Labels[tf]);
                }

                return true;
            }
            catch (Exception)
            {
                Console.WriteLine("no face in database, please add at least a face.");
                return false;
            }
        }



        // train the face
        // return: true if face detected and added
        public Image<Gray, byte> TrainFace(Image<Bgr, Byte> currentFrame, string givenName)
        {
            try
            {
                //Get a gray frame from capture device
                //Image<Gray, byte> gray = _capture.QueryGrayFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                Image<Gray, byte> gray = currentFrame.Convert<Gray, Byte>().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

                //Face Detector
                MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
                _haarFace,
                1.2,
                10,
                Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(20, 20));

                //Action for each element detected
                Image<Gray, byte> TrainedFace = null;
                foreach (MCvAvgComp f in facesDetected[0])
                {
                    TrainedFace = gray.Copy(f.rect).Convert<Gray, byte>();
                    break;
                }

                //resize face detected image for force to compare the same size with the 
                //test image with cubic interpolation type method
                TrainedFace = TrainedFace.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

                // add
                {
                    trainingImages.Add(TrainedFace);
                    labels.Add(givenName);
                }

                // save to file
                {
                    //Write/rewrite the number of triained faces in a file text for further load
                    File.WriteAllText(this.path2db + "/TrainedFaces/TrainedLabels.txt", trainingImages.ToArray().Length.ToString() + "%");

                    //Write the labels of triained faces in a file text for further load
                    for (int i = 1; i < trainingImages.ToArray().Length + 1; i++)
                    {
                        trainingImages.ToArray()[i - 1].Save(this.path2db + "/TrainedFaces/face" + i + ".bmp");
                        File.AppendAllText(this.path2db + "/TrainedFaces/TrainedLabels.txt", labels.ToArray()[i - 1] + "%");
                    }
                }

                return TrainedFace;
            }
            catch(Exception)
            {
                Console.WriteLine("training fail: enable the face detection first.");
                return null;
            }
        }


        // face recognition
        // return:  names for identified faces, also draw rectangle on currentFrame
        public List<string> RecognizeFace(ref Image<Bgr, Byte> currentFrame)
        {
            List<string> names = new List<string>();

            //Get the current frame form capture device
            //currentFrame = _capture.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            currentFrame = currentFrame.Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);  // need ref??

            //Convert it to Grayscale
            Image<Gray, byte> gray = currentFrame.Convert<Gray, Byte>();

            //Face Detector
            MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
                _haarFace, 
                1.2, 
                10,
                Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(20, 20));

            //Action for each element detected
            foreach (MCvAvgComp f in facesDetected[0])
            {
                Image<Gray, byte> faceFrame = currentFrame.Copy(f.rect).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                //draw the face detected in the 0th (gray) channel with blue color
                currentFrame.Draw(f.rect, new Bgr(Color.Red), 2);

                if (trainingImages.ToArray().Length != 0)
                {
                    //TermCriteria for face recognition with numbers of trained images like maxIteration
                    int maxIteration = trainingImages.ToArray().Length;
                    MCvTermCriteria termCrit = new MCvTermCriteria(maxIteration, 0.001);

                    //Eigen face recognizer
                    EigenObjectRecognizer recognizer = new EigenObjectRecognizer(
                       trainingImages.ToArray(),
                       labels.ToArray(),
                       3000,
                       ref termCrit);

                    string name = recognizer.Recognize(faceFrame);
                    if (!string.IsNullOrEmpty(name))
                    {
                        //Draw the label for each face detected and recognized
                        MCvFont font = new MCvFont(FONT.CV_FONT_HERSHEY_TRIPLEX, 0.5d, 0.5d);
                        currentFrame.Draw(name, ref font, new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.LightGreen));

                        names.Add(name);
                    }
                }

                // the others are unknown faces

                /*
                //Set the region of interest on the faces
                        
                gray.ROI = f.rect;
                MCvAvgComp[][] eyesDetected = gray.DetectHaarCascade(
                   _haarEye,
                   1.1,
                   10,
                   Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                   new Size(20, 20));
                gray.ROI = Rectangle.Empty;

                foreach (MCvAvgComp ey in eyesDetected[0])
                {
                    Rectangle eyeRect = ey.rect;
                    eyeRect.Offset(f.rect.X, f.rect.Y);
                    currentFrame.Draw(eyeRect, new Bgr(Color.Blue), 2);
                }
                 */

            }

            return names;
        }


    }
}