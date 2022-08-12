[KO](README_ko.md)

yarn(thread) defect detection CV model 

training / inferring / anomaly score visualization code

My Initial Assumption : If I train an autoencoder with a lot of normal thread image, then inferring an anomaly thread image will result in increase in anomaly score (error)

But the Result : nothing changes. anomaly score is still even whether the image is normal or not

Possible Reason : yarn defect does not make reasonable amount of change within ROI

Possible Next Experiment and Comments : I could use segmentation tool to make ROI in shape of polygon or polyline, instead of current box shape. but the yarn is moving within ROI, so there should be additional detection module to detect thread. Not so good for realtime edge device usecase.