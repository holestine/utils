from chatgpt import q_with_data

resume = f"""
Deep Learning Software Engineer & Tech Lead
Experienced Software Professional with diverse background specializing in Deep Learning and Computer Vision. Designs, implements and improves software systems. Collaborates with analysts, designers & scientists, tests applications, writes training documentation and leads junior engineers.

Experience
Intel Corporation										              Hillsboro, OR
AI Algorithms Engineer										   08/2021  04/2023
Deep Learning Data Scientist 									   10/2020  08/2021
Sr. Machine Learning Engineer 									   02/2017  10/2020
Sr. Software Engineer										   11/2007  02/2017
Application & UI Engineer									   08/2004  11/2007
3D Object Detection
Evaluated a multi-modal deep fusion model for use in a robotics SDK. Created a development plan to port capabilities to custom AMRs and system provisioning instructions for the development environment. Also produced videos of the decoded output showing consistent multi object class identification and location in 3D space using LiDAR and camera sensors.
Feature Matching
Created a novel self-supervised approach to train interest points for monocular visual odometry applications that doesn't require pretraining with synthetic data or the offline creation of a dataset as in the SuperPoint model and does so with a tenth of the parameters in a custom U-Net type model while obtaining HPatches scores competitive with state-of-the-art techniques.
Volumetric Video
Extended the Deep High Resolution Net model so it could be used for inference in a volumetric video pipeline that reconstructs the 3D coordinates of human skeletons using multiple 2D synchronized video streams with different perspectives of a scene. Required transfer learning to extend the model while keeping the relevant pretrained weights and retraining the network on a superset of data including CrowdPose and some proprietary datasets to obtain additional joint locations. Hosted demo at CVPR 2019.
ML Research & Fab Automation
Created a framework in Python used to analyze data collected in the fab during wafer test that utilizes a variety of machine learning algorithms in scikit-learn including Support Vector Classifiers, K-Nearest Neighbors, Random Forest and K-Means as well as fully connected and Convolutional Neural Networks. Was able to identify a subset of failures that previously would have been sent to the next level of validation and provided guidance for future stages of exploration.
Created a Windows application for visualizing numeric data using WPF, WCF and OxyPlot which I open sourced on GitHub. Converted several thousand lines of Matlab code used to analyze various optical signals received from the photonic chip into C# for purposes of automating the wafer test process. Used third party libraries like Math.Net and DotNumerics for well know algorithms. Documented software behaviors in UML and mentored junior engineers and cross disciplinary contributors.
Mobile World Congress 2016
Led a small team to create a suite of demos for Mobile World Congress that showcase the capabilities of Intel’s 5G wireless network. The demos are web based with user interfaces implemented in HTML, CSS and JavaScript and the back end implemented with Node.js utilizing RESTful web services and Sockets for cross machine communications. 4k 360⁰ videos were sent to a variety of clients by the user from an independent website that displays thumbnail videos as previews using the MediaSource API to show low latencies with dense data. Also assumed the graphic designer role and used the 3D modeling tool Blender to augment existing photorealistic scenes in the UI with new objects.
OpenCV Transparent API Demo
Ported several OpenCV 2.x image processing algorithms to OpenCV 3.0, analyzed with VTune to locate areas that could benefit from GPU acceleration and refactored to use the new OpenCV Transparent API which uses OpenCL internally to achieve the optimization.
Metric Improvement with Visual Analytics
Analyzed the Excel based metric creation and reporting process used for pre-Silicon hardware validation and implemented an alternative solution using Tableau that connects directly to the data and provides richer more interactive data enabling visual analytics while requiring significantly less effort to maintain.
UI and Application Design
Led a small team to develop a tool in C/C++ that packages application and driver installers onto a single media saving millions of dollars annually.
Received conceptual UI designs from internal and 3rd party graphic designers and implemented the behaviors with styles, layouts and custom controls that can easily be used dynamically across applications using a variety of technologies including C#, Java, WPF, XAML, CSS and documented behaviors utilizing UML techniques and custom diagrams.
Designed and automated the build process for a variety of projects in order to remove unnecessary dependencies between development teams and track integration results of the source code and binaries.
Research
Organize and run a forum to bring software engineers together and educate on various techniques and tools used in developing Artificial Intelligences focused on architectures and training techniques for Neural Networks. Hosted a study group on Deep Reinforcement Learning based on the book Deep Reinforcement Learning Hands-On.
Mathematic techniques related to computer graphics and scientific visualization (Quaternions, Runge-Kutta integration approximation, Linear Algebra, Differential Geometry, Numeric Calculus, Math Methods for Physicists). 
Created visualizations of the Lorenz Attractor and Euler Spiral using VTK and have developed similar prototypes on Android using OpenGL ES, Windows using WPF 3D and the web using three.js. 
Created a small world in Unity and integrated with the Oculus Rift VR headset to determine feasibility of implementing a novel approach to augmented reality with a perceptual computing camera.

Additional Experience
Portland State University				Graduate Teaching Assistant		              Portland, OR
Analyzed and improved an Evolutionary Algorithm used to optimize the layout of an IC prior to its presentation at the IEEE Congress on Evolutionary Computation.
Developed a technique and implemented software to evolve neural network controlled virtual bipeds to perform locomotion in artificial environments governed by physical rules I defined analytically. 
Tutored/Instructed students in a variety of areas including Calculus, Linear Algebra, Microcontroller Programing (PIC, ARM), Evolutionary Algorithms and Neural Networks.
Micro Systems Engineering Inc. / Biotronik		Software Intern				      Lake Oswego, OR
Developed automated solutions and engineering tools for conducting a variety of stability tests on life-saving circuits (pacemakers, defibrillators) and implemented supporting platform independent code in primarily Java & C++.

Education and Professional Development
Portland State University									              Portland, OR
Master of Science (M.S.) in Electrical & Computer Engineering	                                                                                      2004
Bachelor of Science (B.S.) in Computer Engineering	                                                                                                     2001
Udacity													           Online
Natural Language Processing Nanodegree                                                                                                                               2019
Deep Reinforcement Learning Nanodegree	                                                                                                                   2018

Volunteer
PSU Physics Department Co-Instructor for PH434/534 Math Methods for Physicists			              2018
Orenco Elementary Classroom Assistant									  2015/17/18
Hillsboro School District Chess Coach									              2016
Namibia, Africa Rural Schools Technology Enhancements						              2013

"""
question = 'Create an ordered list of the best jobs for a person with this resume?'
response = q_with_data(question, resume)
print(response)
