# Lightweight 3D Person Tracking

​																				  McClain Thiel

------

###### Abstract

Object detection and localization has historically required two or more sensors due to the loss of information from 3D to 2D space, however, most surveillance systems in use in the world only have one sensor per location. Generally, this consists of a single low-resolution camera positioned above the area under observation (mall, jewelry store, traffic camera). This is not sufficient for robust 3D tracking for applications such as security or more recent relevance, contract tracing. This paper proposes a lightweight system for 3D person tracking that requires no additional hardware, that is based on compressed object detection cov-nets, facial landmark detection, and projective geometry. Preliminary testing, although severely lacking, suggests reasonable success in  3D tracking idea conditions.  

###### Introduction

The need for tracking falls into 3 main Catagories: analytics, anti-theft / auto-checkout, and contact tracing. Brick and mortar retail stores have lacked the complex analytics available to online retailers for years, which has put them at a significant disadvantage. Online retailers have a wealth of information in the form of A/B testing, mouse tracking and click analysis, and infinite data mining techniques which they effectively use to drive sales. The ability to collect more data on consumers without major hardware improvements would provide retailers with substantial values. In addition, anti-theft and auto checkout applications like Amazon's Go store in Seattle would be able to increase the number of locations significantly faster with the ability to track people without installing specialized hardware.  

Additionally, with the rise of COVID-19, the need for quickly deployable person tracking systems in contact tracing application is higher than ever. Single surveillance cameras already populate most public high-density areas such as universities, major cities, and retail locations. Using the massive,  already-deployed surveillance system,  available in most high-risk areas,  would massively increase government and NGO ability to monitor and trace potential viral spread.

$$\rho \text{ - the distance from the camera to the target, } \theta \text{ - the polar angle and, } \phi \text{ - the azimuthal angle}$$









###### Approach



###### Results



###### Conclusion 

###### References 

Merriman, Kendall. “Real-Time 3d Person Tracking and Dense Stereo Maps Using GPU Acceleration.” doi:10.15368/theses.2014.42.

D'Seas, Gregory, et al. “(Faster) Facial Landmark Detector with Dlib.” *PyImageSearch*, 18 Apr. 2020, www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/.

“IMDB-WIKI – 500k+ Face Images with Age and Gender Labels.” *IMDB-WIKI - 500k+ Face Images with Age and Gender Labels*, data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.

Imdeepmind. “Imdeepmind/Processed-Imdb-Wiki-Dataset.” *GitHub*, github.com/imdeepmind/processed-imdb-wiki-dataset.

 I. Everts, G. Jones, and N. Sebe. Cooperative object tracking with multiple ptz cameras. Image Analysis and Processing, 2007.



https://arxiv.org/pdf/1706.06969.pdf



https://www.semanticscholar.org/paper/One-millisecond-face-alignment-with-an-ensemble-of-Kazemi-Sullivan/d78b6a5b0dcaa81b1faea5fb0000045a62513567



list of perturbations: 

- flip
- zoom
- rotate
- scale
- shift
- rotate
- sheer
- brightness
- dropoout
- partial delete
- mesh
- blur



