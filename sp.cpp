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


int main()
{
    vector<map<int,double> > data;
    ifstream file("ionosphere.data.txt");
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
        y.push_back(stoi(tokens[m-1]));

        for (int i = 0; i<m-1; i++)
        {
            if (stod(tokens[i])!=0.0)
            {
                example[i+1] = stod(tokens[i]);
            }
            
        }
        data.push_back(example);
        map<int,double>::iterator it=data[0].begin();
    }
    
    ofstream myfile;
    myfile.open ("tr.txt");
    myfile << "Writing this to a file.\n";
    for (int i=0; i<n;i++)
    {
        myfile<<y[i]<<",";
        for (map<int,double>::iterator it=data[i].begin(); it!=data[i].end(); ++it)
        {
            myfile<<it->first;
            myfile<<":";
            myfile<<it->second;
            if(++it != data[i].end())
            {
                myfile<<",";
                it--;
            }
            if(it==data[i].end())
                it--;
        }
        myfile<<"\n";
    }
    
    
    file.close();
    myfile.close();
    return 0;

}













