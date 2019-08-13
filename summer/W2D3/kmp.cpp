#include<cstdio>
#include<iostream>
#include<cstring>
#include<string>
using namespace std;
int main()
{
    string m,p;
    cin>>p>>m;
    int f[1000];
    memset(f,0,sizeof(f));
    f[0]=-1;
    int j=-1,i=0;
    while(i<m.size())
    {
        while(j!=-1&&m[i]!=p[i])
        {
            j=f[j];
        }
        f[++i]=++j;
    }
    j=0;
    for(int i=0;i<m.size();i++)
    {
        
    }
}