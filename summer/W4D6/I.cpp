#include <cstdio>
#include <iostream>
#include <cstring>
#include <map>
#include <vector>
using namespace std;
typedef long long ll;
const int N = 100050;
map<pair<int, int>, int> m;
vector<int> p[N];
int f[N], n;
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        int a, b;
        scanf("%d%d", &a, &b);
        int l = a + 1, r = n - b;
        if (l > r)
            continue;
        pair<int, int> tmp;
        tmp.first = l, tmp.second = r;
        m[tmp]++;
        if (m[tmp] == 1)
            p[r].push_back(l);
    }
    for (int i = 1; i <= n; i++)
    {
        f[i] = f[i - 1];
        for (int j = 0; j < p[i].size(); j++)
        {
            int k = p[i][j];
            pair<int, int> tmp;
            tmp.first = k, tmp.second = i;
            f[i] = max(f[i], f[k - 1] + min(m[tmp], i - k + 1));
        }
    }
    printf("%d\n", n - f[n]);
}