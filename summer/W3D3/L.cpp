#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
using namespace std;
const int INF = 10000;
const int N = 110;
int l, d;
struct node
{
    int x, y;
};
node men[N], house[N];
int nm, nh;
int mp[N][N];
int lx[N], ly[N], link[N], slack[N];
int visx[N], visy[N];
int abs(int x) { return x < 0 ? -x : x; }
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
bool dfs(int x)
{
    visx[x] = 1;
    for (int i = 1; i <= nh; i++)
    {
        if (visy[i])
            continue;
        int temp = lx[x] + ly[i] - mp[x][i];
        if (temp == 0)
        {
            visy[i] = 1;
            if (link[i] == -1 || dfs(link[i]))
            {
                link[i] = x;
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
    memset(link, -1, sizeof(link));
    for (int i = 1; i <= nm; i++)
    {
        lx[i] = -INF;
        for (int j = 1; j <= nh; j++)
        {
            lx[i] = max(lx[i], mp[i][j]);
        }
    }
    for (int i = 1; i <= nm; i++)
    {
        for (int j = 1; j <= nh; j++)
            slack[j] = INF;
        while (1)
        {
            memset(visx, 0, sizeof(visx));
            memset(visy, 0, sizeof(visy));
            if (dfs(i))
                break;
            int d = INF;
            for (int j = 1; j <= nh; j++)
            {
                if (visy[j])
                    continue;
                d = min(d, slack[j]);
            }
            for (int j = 1; j <= nm; j++)
            {
                if (visx[j])
                {
                    lx[j] -= d;
                }
            }
            for (int j = 1; j <= nh; j++)
            {
                if (visy[j])
                    ly[j] += d;
                else
                    slack[j] -= d;
            }
        }
    }
}
int main()
{
    while (scanf("%d%d", &d, &l))
    {
        memset(mp, 0, sizeof(mp));
        if (d == 0 && l == 0)
            return 0;
        nm=0;
        nh=0;
        for (int i = 1; i <= d; i++)
        {
            string s;
            cin >> s;
            for (int j = 0; j < s.size(); j++)
            {
                if (s[j] == 'm')
                {
                    men[++nm].x = i;
                    men[nm].y = j + 1;
                }
                if (s[j] == 'H')
                {
                    house[++nh].x = i;
                    house[nh].y = j + 1;
                }
            }
        }
        for (int i = 1; i <= nm; i++)
        {
            for (int j = 1; j <= nh; j++)
            {
                mp[i][j] = abs(men[i].x - house[j].x) + abs(men[i].y - house[j].y);
                mp[i][j] = -mp[i][j];
            }
        }
        KM();
        int res[N];
        for (int i = 1; i <= nm; i++)
        if (link[i] != -1)
            res[link[i]] = i;
        int ans = 0;
        for (int i = 1; i <= nm; i++)
        {
            ans += -mp[i][res[i]];
        }
        printf("%d\n", ans);
    }
}