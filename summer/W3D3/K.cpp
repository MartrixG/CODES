#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>
#define ll long long
#define eps 1e-8
#define INF 1999122700000.1912
using namespace std;
int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }
int sgn(double x)
{
    if (-eps < x && x < eps)
        return 0;
    if (x <= -eps)
        return -1;
    else
        return 1;
}
int n, nx, ny;
int lk[104], ans[104];
double lx[104], ly[104], slack[104];
bool visx[104], visy[104];
struct point
{
    double x, y;
    void read()
    {
        scanf("%lf", &x);
        scanf("%lf", &y);
    }
} pa[104], pt[104];
double dist(point A, point B)
{
    return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
}
double mp[104][104];

bool dfs(int x)
{
    visx[x] = 1;
    for (int i = 1; i <= ny; i++)
    {
        if (visy[i])
            continue;
        double temp = lx[x] + ly[i] - mp[x][i];
        if (sgn(temp) == 0)
        {
            visy[i] = 1;
            if (lk[i] == -1 || dfs(lk[i]))
            {
                lk[i] = x;
                return 1;
            }
        }
        else
        {
            slack[i] = min(slack[i], temp);
        }
    }
    return 0;
}

void KM()
{
    memset(lx, 0, sizeof(lx));
    memset(ly, 0, sizeof(ly));
    memset(lk, -1, sizeof(lk));
    nx = ny = n;
    for (int i = 1; i <= nx; i++)
    {
        lx[i] = -INF;
        for (int j = 1; j <= ny; j++)
        {
            lx[i] = max(lx[i], mp[i][j]);
        }
    }
    for (int i = 1; i <= nx; i++)
    {
        for (int j = 1; j <= ny; j++)
            slack[j] = INF;
        while (1)
        {
            memset(visx, 0, sizeof(visx));
            memset(visy, 0, sizeof(visy));
            if (dfs(i))
                break;
            double d = INF;
            for (int j = 1; j <= ny; j++)
            {
                if (visy[j])
                    continue;
                d = min(d, slack[j]);
            }
            for (int j = 1; j <= nx; j++)
            {
                if (visx[j])
                {
                    lx[j] -= d;
                }
            }
            for (int j = 1; j <= ny; j++)
            {
                if (visy[j])
                    ly[j] += d;
                else
                    slack[j] -= d;
            }
        }
    }
}
int w33ha()
{
    for (int i = 1; i <= n; i++)
        pa[i].read();
    for (int i = 1; i <= n; i++)
        pt[i].read();
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            mp[i][j] = -dist(pa[i], pt[j]);
        }
    }
    KM();
    for (int i = 1; i <= n; i++)
        if (lk[i] != -1)
            ans[lk[i]] = i;
    for (int i = 1; i <= n; i++)
        printf("%d\n", ans[i]);
    return 0;
}
int main()
{
    while (scanf("%d", &n) != EOF)
        w33ha();
    return 0;
}
