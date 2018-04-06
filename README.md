FaceSmart
-fanghui

(box.net)
20140304   latest archive
20130607   人脸识别， 肤色及手势识别， 运动检测
20130607   minor revision
20130611   整合了motion detection以及opencv/emgu版本


Integrated:

关于人脸识别：
参考FaceRecProOV.rar, a multiple face detection and recognition in real time.
但是， Face recognition is simply from Emgu.CV EigenObjectRecognizer class and uses 
its function "Recognize" only. The key C# code is from "Emgu.CV\EigenObjectRecognizer.cs" source.
one can directly refer to Emgu.CV and use such class.


关于肤色及手势识别：
skin detection, hand gesture detection, refer to HandGestureRecognition_Code_and_Video- Copia.rar
一部分文档放在 Google+ community -> FaceWise, 还有hotmail里面I2R的几封邮件。


关于运动检测:
motion detection ( copied from Emgu sample )



过滤条件：
1. 检测到人脸，并且在图片中央
if((r->x + (r->width)/2 > 0.4 * img->width) && (r->x + (r->width)/2 < 0.6 * img->width))  //human central only
		{
			faceRect.x=r->x;
			faceRect.y=r->y;
			faceRect.height = r->height;
			faceRect.width = r->width;
		}


2. 该人脸经harr特征已识别在数据库中

3  检测到人手势或运动 (过滤小movements, overall movement, and face movement)


Open CV: 2.20
Emgu: 2.2.1
haar: haarcascade_frontalface_default.xml