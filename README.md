# 440Project
final project for 440

Please be sure to read this first. After this, please read the paper. After the paper, watch the video, and then look at the preprocessing notebook and then the training/testing notebook. My project is somewhat unconventional and can be difficult to understand without the prerequisite knowledge. I got permission from Professor Fabio Abreu de Santos to do this as my final project. For reference, my final project is based on work I completed at the ISAT lab with my mentor, Professor Nathaniel Blanchard along with some help from an undergraduate and graduate students. We submitted the abstract to the HCII conference and were accepted. I am one of the leads on the paper. The paper itself will be much longer and have more content but for this class, I didn't want to do more than the 4 pages that was requested as that felt more fair in comparison to other projects. I understand that machine learning is just one small component of this project but the professor improved of it anyways. Also, he mentioned it was fine if my paper did not have mathmatical diagrams explaining the models (as they are all classical machine learning models). He also said it was fine that most of my code was written with the help of the research lab and has been reused over and over again (not all of it some of it is completely new). Since I did not have a 440 team for this final project, think of my research lab as my team. Finally, he said it was ok that not all of my code can be run by the TA (and it is fine to run it myself on a video for the TA) You can clone this repo, but do not try to run the code without following the directions.

After you read the paper and watch the video, look at the information below.

Be sure to look at the preprocessing.ipynb file first. While you cannot run it yourself, there is extensive documentation showing the result of running each cell and the output it will be. Still look at it as it clears up a lot of context concerning how we preprocess our data

Unfortunately, most of the code has a lot of setup to actually run any of it. I confirmed with Professor Fabio Abreu De Santos and he said this was fine (since it was a research project). However, I wanted the grader to be able to run something so I included an alternate way to train/test the model on CPS data. This alternate way is inside of Train/Test.ipynb. Please run it through Google Colab (watch the video and it will explain it and show proof of me running it).

There are several example files in here as well. BertCPS is an example of what a finalized BERT file will look like (before train/test). I included a verbal and prosodic file so you can see what both modalities look like. The proof of submission image is included as well. Results.zip includes the final results of all of our training data

Trainer.py is the actual training file we used for final results. This one is impossible for the grader to run since it requires advance setup (Spark among many other technologies to run). I attached it though so the grader can understand how our lab is getting these results without having to run things locally

The other Jupyter Notebook attached is for actually visualizing the data. This one cannot be run either (as it requires specific setup) but it can be looked at again to see how our lab sees our final results for this project


