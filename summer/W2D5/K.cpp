#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
const int inf = 0x7fffffff;
struct node
{
    int sum;
    int l, r;
    node *lc;
    node *rc;
};
int a[100000];
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
void build(node *root, int l, int r)
{
    if (l == r)
    {
        root->lc = NULL;
        root->rc = NULL;
        root->l = root->r = l;
        root->sum = a[l];
    }
    else
    {
        int mid = (l + r) >> 1;
        root->lc = new node;
        root->rc = new node;
        build(root->lc, l, mid);
        build(root->rc, mid + 1, r);
        root->sum = root->lc->sum + root->rc->sum;
        root->l = l;
        root->r = r;
    }
}
int query_sum(node *root, int l, int r)
{
    if (root->r < l || root->l > r)
        return 0;
    if (root->l >= l && root->r <= r)
    {
        return root->sum;
    }
    int mid = (l + r) >> 1;
    return query_sum(root->lc, l, r) + query_sum(root->rc, l, r);
}
void modify(node *root, int l, int k)
{
    if (root->l > l || root->r < l)
        return;
    if (root->l == l && root->r == l)
    {
        root->sum += k;
        return;
    }
    modify(root->lc, l, k);
    modify(root->rc, l, k);
    root->sum = root->lc->sum + root->rc->sum;
}
int T;
int main()
{
    scanf("%d\n", &T);
    int cnt = 0;
    while (T--)
    {
        cnt++;
        printf("Case %d:\n", cnt);
        int n;
        scanf("%d", &n);
        for (int i = 1; i <= n; i++)
        {
            scanf("%d", &a[i]);
        }
        node ROOT;
        build(&ROOT, 1, n);
        string s;
        while (cin >> s)
        {
            if (s[0] == 'E')
                break;
            if (s[0] == 'Q')
            {
                int l, r;
                scanf("%d%d", &l, &r);
                printf("%d\n", query_sum(&ROOT, l, r));
            }
            if (s[0] == 'A')
            {
                int l, k;
                scanf("%d%d", &l, &k);
                modify(&ROOT, l, k);
            }
            if (s[0] == 'S')
            {
                int l, k;
                scanf("%d%d", &l, &k);
                modify(&ROOT, l, -k);
            }
        }
    }
}