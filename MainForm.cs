// Online face detection and recognition
// Using EmguCV cross platform .Net wrapper to the Intel OpenCV image processing library for C#.Net

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.Threading;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.VideoSurveillance;
using Emgu.Util;

using HandGestureRecognition;
using HandGestureRecognition.SkinDetector;

namespace FaceSmart
{
    public partial class Form1 : Form
    {
        //camera capture
        Capture _capture;
        //Thread _captureThread;
        //Image<Bgr, Byte> currentFrame;
        
        bool bEnableFaceDetection = true;
        bool bEnableMotionDetection = true;
        bool bEnableHandGestureDetection = false;

        FaceRecognizer face = new FaceRecognizer();

        HandGestureRecognizer hand = new HandGestureRecognizer(); //@".\sample.MPG");

        MotionDetector motion = new MotionDetector();

        bool bAuthorized = false;
        bool bFileOpened = false;
        string secureFileName = Application.StartupPath + "/demo.docx";


        public Form1()
        {
            InitializeComponent();

            button1.Enabled = false;

            if (!face.LoadFace(Application.StartupPath))
            {
                //MessageBox.Show(ex.ToString());
                MessageBox.Show("no face in database, please add at least a face.", "Triained faces load", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }

            //Initialize the capture device
            try
            {
                _capture = new Capture();
                //_capture.QueryFrame();
            }
            catch (Exception)
            {
                MessageBox.Show("no camera captured.");
                return;
            }

            //Initialize the FrameGraber event
            Application.Idle += ProcessImage;
            //Application.Idle += new EventHandler(ProcessImage);
            //_captureThread = new Thread(ProcessImage);
            //_captureThread.Start();
        }


        // try to open document
        private void button1_Click(object sender, EventArgs e)
        {
            // determine if we can open the secured file
            if (!bFileOpened)
            {
                System.Diagnostics.Process.Start(secureFileName);
                bFileOpened = true;
            }
        }

        // train the face
        private void button2_Click(object sender, System.EventArgs e)
        {
            //Get a gray frame from capture device
            Image<Bgr, Byte>  currentFrame = _capture.QueryFrame();

            Image<Gray, byte> TrainedFace = face.TrainFace(currentFrame, textBox1.Text);
            if (TrainedFace != null)
            {
                //Show face added in gray scale
                imageBox1.Image = TrainedFace;

                MessageBox.Show(textBox1.Text + "´s face detected and added :)", "Training OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                MessageBox.Show("Enable the face detection first", "Training Fail", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }


        void ProcessImage(object sender, EventArgs e)
        {
            label3.Text = "0";
            label4.Text = "";

            using (Image<Bgr, Byte> currentFrame = _capture.QueryFrame())
            {
                //Get the current frame form capture device
                Image<Bgr, Byte>  image = currentFrame.Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                //Image<Bgr, Byte> currentFrame = _capture.QuerySmallFrame().PyrUp(); //reduce noise from the image

                bool bHandDetected = false;
                if (bEnableHandGestureDetection)
                {
                    bHandDetected = hand.Recognize(currentFrame);
                    //currentFrame = hand.currentFrame;

                    if (bHandDetected)
                    {
                        label3.Text = "hands on";
                    }
                }

                bool bMovementDetected = false;
                if (bEnableMotionDetection)
                {
                    int cnt = motion.DetectMotion(ref image);

                    //Display the amount of motions found on the current image
                    label3.Text = cnt.ToString();

                    bMovementDetected = (cnt > 0);
                }


                bool bFaceDetected = false;
                if (bEnableFaceDetection)
                {
                    List<string> result = face.RecognizeFace(ref image);
                    string names = "";
                    if (result.Count != 0)
                    {
                        bFaceDetected = true;
                        foreach (string xx in result) { names = ((names == "") ? xx : names + "," + xx); }
                    }

                    label4.Text = names;
                }

                //Show the faces procesed and recognized
                imageBoxFrameGrabber.Image = image;
                bAuthorized = (bFaceDetected && (!bHandDetected) && (!bMovementDetected));
            }

            button1.Enabled = bAuthorized;
            if (bFileOpened && !bAuthorized)
            {
                string wordprocess = "WINWORD";
                Process[] procs = Process.GetProcessesByName(wordprocess);
                if (procs.Length > 0)
                {
                    Process proc = procs[0];
                    proc.Kill();
                    proc.WaitForExit();
                }
                bFileOpened = false;
            }
        }

        // call external functions
        private void button3_Click(object sender, EventArgs e)
        {
            //Process.Start("link.html");
        }


    }
}