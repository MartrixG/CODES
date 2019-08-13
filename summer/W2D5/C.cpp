#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <algorithm>
#include <queue>
using namespace std;
int n;
struct d
{
    double x1, x2, x3, x4;
    double y1, y2, y3, y4;
    void set(double a, double b, double c, double d)
    {
        x1 = a, y1 = b;
        x3 = c, y3 = d;
        x2 = x3, y2 = y1;
        x4 = x1, y4 = y3;
    }
};
struct line
{
    int d;
    double hight;
    const bool operator<(const line o) const
    {
        return hight > o.hight;
    }
};
d seq[101];
double X[210];
int main()
{
    int cnt = 0;
    while (scanf("%d", &n))
    {
        if (n == 0)
            return 0;
        cnt++;
        for (int i = 1; i <= n; i++)
        {
            double a, b, c, d;
            scanf("%lf%lf%lf%lf", &a, &b, &c, &d);
            seq[i].set(a, b, c, d);
            X[i] = a;
            X[i + n] = c;
        }
        sort(X + 1, X + n + n + 1);
        int tot = unique(X + 1, X + n + n + 1) - (X + 1);
        double ans = 0;
        for (int i = 1; i <= tot - 1; i++)
        {
            double x1 = X[i], x2 = X[i + 1];
            priority_queue<line> y;
            for (int j = 1; j <= n; j++)
            {
                if (seq[j].x1 <= x1 && seq[j].x2 >= x2)
                {
                    line down, up;
                    down.d = 1, up.d = -1;
                    down.hight = seq[j].y1, up.hight = seq[j].y4;
                    y.push(down);
                    y.push(up);
                }
            }
            int f = 0;
            double w = x2 - x1;
            if (y.empty())
                continue;
            double last = y.top().hight;
            while (!y.empty())
            {
                line top = y.top();
                y.pop();
                if (f > 0)
                {
                    ans += w * (top.hight - last);
                }
                f += top.d;
                last = top.hight;
            }
        }
        printf("Test case #%d\n", cnt);
        printf("Total explored area: %.2f\n\n", ans);
    }
}