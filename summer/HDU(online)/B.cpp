#include <cstdio>
#include <set>
#include <iostream>
using namespace std;
const int N = 1000010;
int T;
int a[N];
int main()
{
    scanf("%d", &T);
    while (T--)
    {
        int n, m;
        scanf("%d%d", &n, &m);
        set<int> s;
        for (int i = 1; i <= n; i++)
        {
            scanf("%d", &a[i]);
            s.insert(a[i]);
        }
        int last = 0, l = 1;
        for (int i = 1; i <= m; i++)
        {
            int op;
            scanf("%d", &op);
            if (op == 1)
            {
                int pos;
                scanf("%d", &pos);
                pos ^= last;
                if (s.count(a[pos]) == 1)
                {
                    s.erase(a[pos]);
                }
                a[pos] += 10000000;
                s.insert(a[pos]);
            }
            if (op == 2)
            {
                int r, k;
                scanf("%d%d", &r, &k);
                r ^= last;
                r++;
                k ^= last;
                for (int i = l; i < r; i++)
                {
                    s.erase(a[i]);
                }
                l = r;
                last = *s.lower_bound(k);
                printf("%d\n", last);
            }
        }
    }
}
/*
3
5 9
4 3 1 2 5 
2 1 1
2 2 2
2 6 7
2 1 3
2 6 3
2 0 4
1 5
2 3 7
2 4 3
10 6
1 2 4 6 3 5 9 10 7 8 
2 7 2
1 2
2 0 5
2 11 10
1 3
2 3 2
10 10
9 7 5 3 4 10 6 2 1 8 
1 10
2 8 9
1 12
2 15 15
1 12
2 1 3
1 9
1 12
2 2 2
1 9

*/