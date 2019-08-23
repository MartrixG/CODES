#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
long long t;
long long aa[35], bb[35];
int main()
{
    scanf("%lld", &t);
    while (t--)
    {
        long long a, b;
        scanf("%lld%lld", &a, &b);
        memset(aa, 0, sizeof(aa));
        memset(bb, 0, sizeof(bb));
        long long tot = 0;
        while (a)
        {
            aa[++tot] = a & 1ll;
            a >>= 1ll;
        }
        tot = 0;
        while (b)
        {
            bb[++tot] = b & 1ll;
            b >>= 1ll;
        }
        long long c = 0;
        for (long long i = 1ll; i <= 32; i++)
        {
            if (aa[i] == 1ll && bb[i] == 1ll)
            {
                c += (1ll << (i - 1ll));
            }
        }
        printf("%lld\n", c);
    }
}
//4294967295‬