// L1-regularized logistic regression implementation using stochastic gradient descent

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <string>


using namespace std;

vector<string> split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void usage(const char* prog){

   cout << "L1-regularized logistic regression implementation using stochastic gradient descent:\n" << endl;
   cout << "Options:" << endl; 
   cout << "-s <int>    Shuffle dataset after each iteration. default 1, and 0 indicates no Shuffle" << endl;
    cout<< "-sp <bool>  The data file is stored as spase format. default 1"<<endl;
   cout << "-i <int>    Maximum iterations. default 50000" << endl;   
   cout << "-e <float>  Convergence rate. default 0.005, larger than 1 not recommended" << endl;    
   cout << "-l <float>  L1 regularization weight. default 0.0001" << endl;
   cout << "-lr <float> Select adaptive learning rate methods,default 0 - exponential decay"<<endl;
   cout << "-a <float>  Learning rate. default 0.001" << endl;
   cout << "-m <file>   Read weights from file" << endl;
   cout << "-o <file>   Write weights to file" << endl;
   cout << "-tr <file>  Train file from source" << endl;  
   cout << "-t <file>   Test file to classify" << endl;     
   cout << "-p <file>   Write predictions to file" << endl;
   cout << "-sc         Scale the data to the range of 0 to 1" << endl;     
   cout << "-r          Randomise weights between -1 and 1, otherwise 0" << endl;    
   cout << "-v          Verbose, showing more information" << endl << endl;      
}

double vecnorm(map<int,double>& w1, map<int,double>& w2){

    double sum = 0.0;
    for(auto it = w1.begin(); it != w1.end(); it++){
        double minus = w1[it->first] - w2[it->first];
        double r = minus * minus;
        sum += r;
    }
    return sqrt(sum);
}

double l1norm(map<int,double>& weights){

    double sum = 0.0;
    for(auto it = weights.begin(); it != weights.end(); it++){
        sum +=  fabs(it->second);
    }
    return sum;
}


double sigmoid(double x){

    static double overflow = 20.0;
    if (x > overflow) x = overflow;
    if (x < -overflow) x = -overflow;

    return 1.0/(1.0 + exp(-x));
}

double classify(map<int,double>& features, map<int,double>& weights){

    double logit = 0.0;
    for(auto it = features.begin(); it != features.end(); it++){
        if(it->first != 0){
            logit += it->second * weights[it->first];
        }
    }
    return sigmoid(logit);
}

void normalize(vector<map<int, double> > &data, int p){
            vector<double> min;
            vector<double> max;
            int countzero = 0;
            for(int i=1; i<=p; i++){
                set<double> temp;
                countzero = 0;
                for(int j=0; j<data.size(); j++){
                    if(data[j][i] == 0 && countzero <1){countzero++;temp.insert(data[j][i]);}
                    if(data[j][i] !=0){
                        temp.insert(data[j][i]);
                    }
                }
                set<double> ::iterator it1 = temp.begin();
                set<double> ::iterator it2 = temp.end();
                --it2;
                min.push_back(*it1);
                max.push_back(*it2);
            }
            for(int i=1; i<=p; i++){
                for(int j=0; j<data.size(); j++){
                    if(((max[i-1] !=0 ) || (min[i-1]!=0 )) && data[j][i] !=0 )
                        data[j][i] = (data[j][i] - min[i-1]) / (max[i-1] - min[i-1]);;
                }
                //if(i%1000 == 0){
                    //cout <<"finished running 1000 lines" <<endl;
                //}
            }

            for(int j=0; j<data.size(); j++){
                for(int i=1; i<=p; i++){
                    if(data[j][i] == 0){
                        data[j].erase(i);
                    }
                }
            }
}

void readWeights(ifstream& fin, string line, map<int, double> &weights){
    while (getline(fin, line)){
        if(line.length()){
            if(line[0] != '#' && line[0] != ' '){
                vector<string> tokens2 = split(line,' ');
                if(tokens2.size() == 2){
                    weights[atoi(tokens2[0].c_str())] = atof(tokens2[1].c_str());
                }
            }
        }
    }
    if(!weights.size()){
        cout << "Failed to read weights from file!" << endl;
        fin.close();      
        exit(-1);
    }fin.close();
}

void trainingWeights(ifstream &fin, string line,vector<map<int, double> > &data,map<int,double> &weights,map<int,double> &total_l1,
    int randw){
        random_device rd;
    
        while (getline(fin, line)){
            if(line.length()){
                if(line[0] != '#' && line[0] != ' '){
                    vector<string> tokens = split(line,' ');
                    map<int,double> example;
                    if(atoi(tokens[0].c_str()) == 1){
                        example[0] = 1;
                    }else{
                        example[0] = 0;
                    }
                    for(unsigned int i = 1; i < tokens.size(); i++){
                        //if(strstr (tokens[i],"#") == NULL){
                            vector<string> feat_val = split(tokens[i],':');
                            if(feat_val.size() == 2){
                                example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                                if(randw){
                                    weights[atoi(feat_val[0].c_str())] = -1.0+2.0*(double)rd()/rd.max();
                                }else{
                                    weights[atoi(feat_val[0].c_str())] = 0.0;
                                }
                                total_l1[atoi(feat_val[0].c_str())] = 0.0;
                            }
                        //}
                    }
                    data.push_back(example);
                    //if(verbose) cout << "read example " << data.size() << " - found " << example.size()-1 << " features." << endl; 
                }    
            }
        }
    }

void SGD(vector<map<int,double> >&data, map<int, double> &weights,map<int, double>&total_l1,
 double &alpha, double lro, double maxit,double eps,double norm,int n, double &mu,mt19937 g,int shuf,double l1,vector<int> &index){
        while(norm > eps){

            map<int,double> old_weights(weights);
            if(shuf) shuffle(index.begin(),index.end(),g);

            for (unsigned int i = 0; i < data.size(); i++){
                mu += (l1*alpha);
                //cout<<l1<<endl;
                int label = data[index[i]][0];
                double predicted = classify(data[index[i]],weights);
                for(auto it = data[index[i]].begin(); it != data[index[i]].end(); it++){
                    if(it->first != 0){
                        weights[it->first] += alpha * (label - predicted) * it->second;
                        if(l1){
                            // Cumulative L1-regularization
                            // Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009
                            // http://aclweb.org/anthology/P/P09/P09-1054.pdf
                            double z = weights[it->first];
                            if(weights[it->first] > 0.0){
                                weights[it->first] = max(0.0,(double)(weights[it->first] - (mu + total_l1[it->first])));
                            }else if(weights[it->first] < 0.0){
                                weights[it->first] = min(0.0,(double)(weights[it->first] + (mu - total_l1[it->first])));
                            }
                            total_l1[it->first] += (weights[it->first] - z);
                        }    
                    }
                }
            }
            //cout<<weights<<endl;
            //cout<<"old"<<old_weights<<endl;
            norm = vecnorm(weights,old_weights);
            if(n && n % 100 == 0){
                double l1n = l1norm(weights);
                printf("# convergence: %1.4f l1-norm: %1.4e iterations: %i\n",norm,l1n,n);     
            }
            if(++n > maxit){
                break;
            }
            //Update learning rate
            if (lro==0)
            {
                alpha = alpha/(1+((int)n+1)/maxit);
            }
            else if(lro!=1 & lro != 0)
            {
                alpha = alpha*pow(lro,((int)n+1)/(float)maxit);
                //cout<<maxit<<endl;
                //cout<<-(n+1)<<endl;
                //cout<<((int)n+1)/(float)maxit<<endl;
                //cout<<lro<<endl;
                //cout<<pow(lro,((double)n+1)/(double)maxit)<<endl;
            }

            
        }
}

void testPredict(ifstream &fin, map<int,double> &weights, int verbose, string line,
    double& tp,double& fp,double& tn,double& fn,string predict_file,ofstream& outfile){        
    while (getline(fin, line)){
        if(line.length()){
            if(line[0] != '#' && line[0] != ' '){
                vector<string> tokens = split(line,' ');
                map<int,double> example;
                int label = atoi(tokens[0].c_str());
                for(unsigned int i = 1; i < tokens.size(); i++){
                    vector<string> feat_val = split(tokens[i],':');
                    example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                }
                double predicted = classify(example,weights);
                if(verbose){
                    if(label > 0){
                        printf("label: +%i : prediction: %1.3f",label,predicted);
                    }else{
                        printf("label: %i : prediction: %1.3f",label,predicted);
                        }
                    }
                if(predict_file.length()){
                    if(predicted > 0.5){
                        outfile << "1" << endl;
                    }else{
                        outfile << "0" << endl;
                    }
                }
                if(((label == -1 || label == 0) && predicted <= 0.5) || (label == 1 && predicted > 0.5)){
                    if(label == 1){tp++;}else{tn++;}    
                    if(verbose) cout << "\tcorrect" << endl;
                }else{
                    if(label == 1){fn++;}else{fp++;}    
                    if(verbose) cout << "\tincorrect" << endl;
                }
            }    
        }
    }
}

void GenerateSparse(string train, int op)
{
    ifstream file(train);
    vector<map<int,double> > dat;
    vector<int> y;
    string str;
    int n = 0;
    int m = 0;
    //cout<<stoi("-1")<<endl;
    while (getline(file, str))
    {
        n++;
        
        map<int,double> example;
        vector<string> tokens = split(str,',');
        
        
        m=tokens.size();
        
        if (stoi(tokens[m-1])==0)
        {
            y.push_back(-1);
        }
        else{
            y.push_back(stoi(tokens[m-1]));
        }
        
        for (int i = 0; i<m-1; i++)
        {
            if (stod(tokens[i])!=0.0)
            {
                example[i+1] = stod(tokens[i]);
            }
            
        }
        dat.push_back(example);
        //map<int,double>::iterator it=dat[0].begin();
    }
    
    
    
    ofstream myfile;
    if (op==0)
    {
    myfile.open ("sparseFormat.txt");
    }
    else
        myfile.open("sparseFormatT.txt");
    //myfile << "Writing this to a file.\n";
    for (int i=0; i<n;i++)
    {
        myfile<<y[i]<<" ";
        for (map<int,double>::iterator it=dat[i].begin(); it!=dat[i].end(); ++it)
        {
            myfile<<it->first;
            myfile<<":";
            myfile<<it->second;
            myfile<<" ";
            //if(++it != dat[i].end())
            //{
              //  myfile<<",";
                //it--;
            //}
            if(it==dat[i].end())
                it--;
        }
        myfile<<"\n";
    }
    
    
    myfile.close();

}


int main(int argc, const char* argv[]){

    //Scale or not
    int scale = 0;
    
    //learning rate options
    double lro = 0;

    double alpha = 0.001; // Learning rate
    double l1 = 0.0001; // L1 penalty weight
    unsigned int maxit = 50000; // Max iterations
    int shuf = 1; // Shuffle data set
    double eps = 0.005; // Convergence threshold
    int verbose = 0; // Verbose
    int randw = 0; // Randomise weights
    bool spar = 1; //default as sparse format
    string model_in = ""; // Read model file
    string model_out = ""; // Write model file
    string test_file = ""; // Test file
    string predict_file = ""; // Predictions file
    string train = "train.dat";
    

    usage(argv[0]);
    string Options;
    //cin.ignore(INT_MAX);
    cout << "Please input options:";
    getline(cin, Options);
    vector<string> tokens = split(Options,' ');
    cout << "# called with:       ";
    for(int i = 0; i < (int)tokens.size(); i++){
        cout << i+1 << " ";
        if(tokens[i] == "-a" ) alpha = stof(tokens[i+1]);
        if(tokens[i] == "-m" ) model_in = string(tokens[i+1]);
        if(tokens[i] == "-o" ) model_out = string(tokens[i+1]);
        if(tokens[i] == "-t" ) test_file = string(tokens[i+1]);
        if(tokens[i] == "-p" ) predict_file = string(tokens[i+1]);
        if(tokens[i] == "-s" ) shuf = stoi(tokens[i+1]);
        if(tokens[i] == "-i" ) maxit = stoi(tokens[i+1]);
        if(tokens[i] == "-tr" ) train = string(tokens[i+1]);
        if(tokens[i] == "-lr") lro = stof(tokens[i+1]);
        if(tokens[i] == "-sc") scale = 1;
        if(tokens[i] == "-sp") spar = stoi(tokens[i+1]);
        if(tokens[i] == "-e" ) eps = stof(tokens[i+1]);
        if(tokens[i] == "-l" ) l1 = stof(tokens[i+1]);
        if(tokens[i] == "-v")  verbose = 1;
        if(tokens[i] == "-r")  randw = 1;
        if(tokens[i] == "-h"){
            usage(argv[0]);
            return(1);
        }
    }
    cout << "        Complete" <<endl << endl;
    

    
    if(!model_in.length()){
        cout << "# Learning Rate update method:    "<<lro<<endl;
        cout << "# Initial learning rate:     " << alpha << endl;
        cout << "# convergence rate:  " << eps << endl;
        cout << "# l1 penalty weight: " << l1 << endl;
        cout << "# max. iterations:   " << maxit << endl;   
        cout << "# training data:     " << argv[argc-1] << endl;
        if(model_out.length()) cout << "# model output:      " << model_out << endl;
    }
    if(model_in.length()) cout << "# model input:       " << model_in << endl;
    if(test_file.length()) cout << "# test data:         " << test_file << endl;
    
    if(spar==0)
    {
        //ifstream fte;
        //fte.open(test_file);
        //ofstream myte;
        //myte.open ("sparseFormatT.txt");
        GenerateSparse(test_file,1);
        //fte.close();

    }
   
    if(predict_file.length()) cout << "# predictions:       " << predict_file << endl;

    vector<map<int,double> > data;
    map<int,double> weights;
    map<int,double> total_l1;
    random_device rd;
    mt19937 g(rd());
    ifstream fin;
    string line;


    // Read weights from model file, if provided
    if(model_in.length()){
        fin.open(model_in.c_str());
        readWeights(fin, line, weights);
    }

    
    // If no weights file provided, read training file and calculate weights
    if(!weights.size()){
        if (!spar)
        {
            GenerateSparse(train,0);
            train="sparseFormat.txt";
        }
        fin.open(train);
        trainingWeights(fin, line, data, weights, total_l1, randw);
            
        fin.close();

        cout << "# training examples: " << data.size() << endl;
        cout << "# features:          " << weights.size() << endl;
        
        if(scale){
            cout << "Normalizing..." <<endl;
            normalize(data,weights.size());
        }
        
        double mu = 0.0;
        double norm = 1.0;
        int n = 0;
        vector<int> index(data.size());
        iota(index.begin(),index.end(),0);

        cout << "# stochastic gradient descent" << endl;
        SGD(data, weights,total_l1, alpha, lro,maxit, eps, norm, n, mu,g,shuf,l1,index);


        unsigned int sparsity = 0;
        for(auto it = weights.begin(); it != weights.end(); it++){
            if(it->second  != 0) sparsity++;
        }
        printf("# sparsity:    %1.4f (%i/%i)\n",(double)sparsity/weights.size(),sparsity,(int)weights.size());     

        if(model_out.length()){
            ofstream outfile;
            outfile.open(model_out.c_str());  
            for(auto it = weights.begin(); it != weights.end(); it++){
                outfile << it->first << " " << it->second << endl;
            }
            outfile.close();
            cout << "# written weights to file " << model_out << endl;
        }

    }

    // If a test file is provided, classify it using either weights from
    // the provided weights file, or those just calculated from training
    if(test_file.length()){

        ofstream outfile;
        if(predict_file.length()){
            outfile.open(predict_file.c_str());  
        }
        cout << endl;
        cout << "Testing set classification:" << endl;

        double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
        if (!spar)
            test_file = "sparseFormatT.txt";
        fin.open(test_file.c_str());
        testPredict(fin, weights, verbose, line, tp, fp, tn, fn,predict_file,outfile);
        fin.close();

        printf ("# accuracy:    %1.4f (%i/%i)\n",((tp+tn)/(tp+tn+fp+fn)),(int)(tp+tn),(int)(tp+tn+fp+fn));
        printf ("# precision:   %1.4f\n",tp/(tp+fp));
        printf ("# recall:      %1.4f\n",tp/(tp+fn));
        printf ("# mcc:         %1.4f\n",((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));
        printf ("# tp:          %i\n",(int)tp);
        printf ("# tn:          %i\n",(int)tn);
        printf ("# fp:          %i\n",(int)fp);    
        printf ("# fn:          %i\n",(int)fn);

        if(predict_file.length()){
            cout << "# written predictions to file " << predict_file << endl;
            outfile.close();
        }    
    }

    return(0);

}
