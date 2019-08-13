#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cmath>
using namespace std;
const int N = 2e5 + 15;
struct node
{
    int lc, rc, l, r, max;
} t[2 * N];
int root, tot = 0;
int a[N];
void build(int x, int l, int r)
{
    t[x].l = l;
    t[x].r = r;
    if (l == r)
    {
        t[x].max = a[l];
        return;
    }
    int mid = (l + r) >> 1;
    t[x].lc = ++tot;
    build(t[x].lc, l, mid);
    t[x].rc = ++tot;
    build(t[x].rc, mid + 1, r);
    t[x].max = max(t[t[x].lc].max, t[t[x].rc].max);
}
int query_max(int x, int l, int r)
{
    if (l <= t[x].l && t[x].r <= r)
        return t[x].max;
    int mid = (t[x].l + t[x].r) >> 1;
    int ans = 0;
    if (l <= mid)
        ans = max(ans, query_max(t[x].lc, l, r));
    if (mid < r)
        ans = max(ans, query_max(t[x].rc, l, r));
    return ans;
}
void modify(int x, int l, int d)
{
    if (t[x].l == t[x].r)
    {
        t[x].max = d;
        return;
    }
    int mid = (t[x].l + t[x].r) / 2;
    if (l <= mid)
        modify(t[x].lc, l, d);
    if (mid < l)
        modify(t[x].rc, l, d);
    t[x].max = max(t[t[x].lc].max, t[t[x].rc].max);
}

int main()
{
    int n, m;
    while (scanf("%d %d", &n, &m) != EOF)
    {
        for (int i = 1; i <= n; i++)
            scanf("%d", &a[i]);
        tot = 0;
        root = ++tot;
        memset(t, 0, sizeof(t));
        build(root, 1, n);
        char c;
        for (int i = 1; i <= m; i++)
        {
            cin >> c;
            int x, y;
            scanf("%d %d", &x, &y);
            if (c == 'Q')
            {
                int ans = query_max(1, x, y);
                printf("%d\n", ans);
            }
            if (c == 'U')
            {
                modify(1, x, y);
            }
        }
    }
}
