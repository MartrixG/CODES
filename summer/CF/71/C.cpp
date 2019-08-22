#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
typedef long long ll;
int T;
int min(int a, int b) { return a < b ? a : b; }
struct duan
{
    ll begin, end;
};
duan d[35];
/*
1
15
3 5
010010001000010
*/
int main()
{
    scanf("%d", &T);
    while (T--)
    {
        memset(d, 0, sizeof(d));
        ll n, a, b;
        scanf("%lld%lld%lld", &n, &a, &b);
        string s;
        cin >> s;
        ll ans = 0;
        ans = (n + 1) * b + n * a;
        int f = 0;
        ll begin = -1;
        int tot = 0;
        for (ll i = 0; i < s.size(); i++)
        {
            if (f == 0)
            {
                if (s[i] == '1')
                {
                    d[++tot].begin = 0;
                    d[tot].end = i - 1;
                    f = 1;
                    begin = i;
                }
            }
            else
            {
                if (s[i] == '1' && begin == -1)
                {
                    begin = i;
                }
                if(begin!=-1)
                {
                    if ((s[i] == '0' && s[i + 1] == '0'))
                    {
                        d[++tot].begin = begin;
                        d[tot].end = i - 1;
                        begin = -1;
                    }
                    if (i + 1 == s.size() - 1)
                    {
                        d[++tot].begin = begin;
                        d[tot].end = i;
                    }
                }
            }
        }
        if (tot != 0)
        {
            ans += min((d[1].end - d[1].begin + 1) * b, a);
            for (int i = 2; i <= tot; i++)
            {
                ans += b * (d[i].end - d[i].begin + 2);
            }
            for (int i = 2; i <= tot - 1; i++)
            {
                ans += min(b * (d[i + 1].begin - d[i].end - 2), 2 * a);
            }
            ans += min(a, b * (s.size() - d[tot].end - 1));
        }
        printf("%lld\n", ans);
    }
}