#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <bitset>
using namespace std;
int n;
bitset <2010> a[2010];
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        a[i][i]=1;
        string s;
        cin >> s;
        for (int j = 0; j < s.size(); j++)
        {
            if (s[j] == '1')
            {
                a[i][j+1]=1;
            }
        }
    }
    int ans = 0;
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<=n;j++)
        {
            if(a[j][i])
                a[j]|=a[i];
        }
    }
    for(int i=1;i<=n;i++)
    {
        ans+=a[i].count();
    }
    printf("%d\n", ans);
}