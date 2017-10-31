# SGD-for-L1-regularized-Log-linear-Models
Stochastic Gradient Descent for L1-regularized Log-linear Models

User Manual

Compile by running ’make’. Uses -std=c++11 - on older compilers you may need to change this to -std=c++0x in the Makefile.

Run all tests with ’make test’.

Run the train/classify tool as follows: ./doraemon

Training and test data should be in svm-light/libsvm format, e.g.

+1 1:4.12069 6:11.3896 18:18.5742 332:2.85764 5284:53.4406

+1 as the target value marks a positive example, -1 a negative example respectively. Feature number 1 has the value 4.12069, feature number 6 has the value 11.3896, feature number 18 has the value 18.5742, feature 332 has 2.85764 and feature 5284 has the value 53.4406 and all the other features have value 0.

Options:
-s <int> Shuffle dataset after each iteration. Default value 1, which is recommended to improve prediction accuracy, and the other possible value 0 indicates no Shuffle.
-i <int> Maximum iterations. Recommend not exceed 10000. Default value 3000.
-e <float> Convergence rate threshold, and the algorithm would stop if error rate difference is smaller than it. Default value 0.005, any extreme small numbers is strongly not recommended.
-a <float> Learning rate in updating feature weights. Default value 0.001, larger than 0.1 not recommended.
-l <float> L1 regularization weight. Recommend not larger than 0.01. Default value 0.0001.
1-lr <float> Select adaptive learning rate methods,default 0 - exponential decay, 1 - constant rate, and other than 0 and 1 will refer one parameter of a strategy of updating learning rate proposed by Collins et al 2008 .
-m <file > Read weights from file. Input the weight file in type string. Default 0.
-o <file> Write weights to file. Output the generated weight file to destination in type string. Default output
-tr <file> Train file from source. Input the training data file in type string. Default file name ”train.dat”
-t <file> Test file to classify. Input the testing data file in type string. Default file name ”test.dat” -p <file> Write predictions to file. Output the generated prediction data file to destination in type string. Default output.
-sp 1:sparse 0: dense. Default 1
-sc Scale the data to the range of 0 to 1. Switch on if added in options flow.
-r Randomize weights between -1 and 1, otherwise 0. Switch on if added in options flow. -v Verbose, showing more information. Switch on if added in options flow.
-h Show the options chart one more time.
Note that, we have two options to generate the learning rate, one is ηk = η0 1+k/N
2008), which is not recommended. The other is ηk = η0αk/N .
(Collins er al.,
 Overall:
Step1: make -f Makefile
Step2: ./doraemon
Step3: choose your options, for example:
-o weights.out -p predict.out -lr 0.85 -i 3000 -e 0.0001 -t test.dat -tr train.dat
if you further want to use normal format datasets, you can add ’-sp 0’ in the options.
-o weights.out -p predict.out -lr 1 -i 3000 -e 0.0001 -tr [YourTrain.txt] -t [YourTest.txt] -sp 0
In order to adapt to different type of matrix (Sparse / Dense), we also provide an option -sp <bool>for switching on dense matrix transformation to sparse matrix form. In the following test- ing sample, 2 sparse samples and 1 dense sample would be presented.
