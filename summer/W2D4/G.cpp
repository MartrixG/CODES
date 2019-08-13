#include<cstdio>
#include<vector>
#include<algorithm>
#include<iostream>
using namespace std;
const int INF=10000000;
int n;
vector<int> tree;
int main()
{
    scanf("%d",&n);
    int op,x;
    int f=0;
    int ans=0;
    for(int i=1;i<=n;i++)
    {
        scanf("%d%d",&op,&x);
        if(tree.size()==0)
        {
            f=op;
            tree.insert(upper_bound(tree.begin(),tree.end(),x),x);
        }
        else if(op==f)
        {
            tree.insert(upper_bound(tree.begin(),tree.end(),x),x);
        }
        else if(op!=f)
        {
            vector<int>::iterator xiao,da;
            xiao=lower_bound(tree.begin(),tree.end(),x);
            if(*xiao==x)
            {
                ans+=0;
                tree.erase(xiao);
                continue;
            }
            else
            {
                xiao--;
            }
            da=upper_bound(tree.begin(),tree.end(),x);
            if(da==tree.begin())
            {
                ans+=(*da)-x;
                tree.erase(da);
            }
            else if(da==tree.end())
            {
                da--;
                ans+=x-(*da);
                tree.erase(da);
            }
            else
            {
                if((*da)-x<x-(*xiao))
                {
                    ans+=(*da)-x;
                    tree.erase(da);
                }
                else
                {
                    ans+=x-(*xiao);
                    tree.erase(xiao);
                }
            }
            ans%=1000000;
        }
    }
    printf("%d\n",ans);
}