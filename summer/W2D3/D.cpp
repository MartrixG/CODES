#include<cstdio>
#include<iostream>
#include<cstring>
#include<string>
#include<cmath>
int min(int a,int b)
{
    if(a<b) return a;
    else return b;
}
int max(int a,int b)
{
    if(a>b) return a;
    else return b;
}
using namespace std;
string s2;
int r[200020];
int ll[200100],rr[200100];
void manacher() {
    int right = 0, pos = -1;
    for (int i = 1; i < s2.size(); i++) {
        int x;
        if (right < i) x = 1;
        else x = min(r[2 * pos - i], right - i);
        while (s2[i - x] == s2[i + x])
        {
            x++;
        }
        if (x + i > right) {
            right = x + i;
            pos = i;
        }
        r[i] = x;
    }
}
int main()
{
    string s;
    cin>>s;
    s2+="%#";
    for(int i=0;i<s.size();i++)
    {
    s2+=s[i];
    s2+='#';
    }
    manacher();
    int n=1;
    for(int i=1;i<s2.size();i+=2)
    {
        while(n+r[n]<i) n++;
        rr[i]=i-n;
    }
    n=s2.size()-1;
    for(int i=s2.size()-1;i>=1;i--)
    {
        while(n-r[n]>i) n--;
        ll[i]=n-i;
    }
    int ans=0;
    for(int i=1;i<s2.size();i++)
    {
        if(ll[i]&&rr[i])
            ans=max(ans,ll[i]+rr[i]);
    }
    printf("%d\n",ans);
}