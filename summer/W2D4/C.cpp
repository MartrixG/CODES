#include<cstdio>
#include<iostream>
#include<set>
using namespace std;
int main()
{
    int n;
    scanf("%d",&n);
    char op;
    int x;
    set<int> s;
    for(int i=1;i<=n;i++)
    {
        getchar();
        scanf("%c%d",&op,&x);
        if(op=='I')
        {
            s.insert(x);
        }
        if(op=='D')
        {
            int y;
            scanf("%d",&y);
            s.erase(s.lower_bound(x),s.upper_bound(y));
        }
        if(op=='Q')
        {
            printf("%d\n",*(--s.upper_bound(x)));
        }
    }
    return 0;
}