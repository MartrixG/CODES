#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
int main()
{
    string s;
    cin >> s;
    int l = s.size();
    int ans = 0;
    if (l % 2 == 0)
    {
        ans = l / 2;
    }
    else
    {
        int f = 0;
        for (int i = 1; i < l; i++)
            if (s[i] == '1')
                f = 1;
        ans = (l - 1) / 2;
        ans += f;
    }
    if (s[0] == '0' && l == 1)
        ans = 0;
    printf("%d\n", ans);
}