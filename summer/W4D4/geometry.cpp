#include <cstdio>
#include <cmath>
#include <iostream>
#include <algorithm>
using namespace std;
//p1p2和p1p3的叉积
#define cross(p1, p2, p3) ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

const double eps = 1e-9;
inline int sign(double a) { return a < -eps ? -1 : a > eps; }
inline int cmp(double a, double b) { return sign(a - b); }
struct P
{
    double x, y;
    P() {}
    P(double _x, double _y) { x = _x, y = _y; }
    double dot(P p) { return x * p.x + y * p.y; }
    double det(P p) { return x * p.y - y * p.x; }
    double mod() { return sqrt(x * x + y * y); }
    P rot(double an) { return {x * cos(an) - y * sin(an), x * sin(an) + y * cos(an)}; }
    P operator+(P p) { return {x + p.x, y + p.y}; }
    P operator-(P p) { return {x - p.x, y - p.y}; }
    P operator*(double d) { return {x * d, y * d}; }
    P operator/(double d) { return {x / d, y / d}; }
    double alpha() { return atan2(y, x); }
    void read() { cin >> x >> y; }
};
//直线平行
bool checkll(P p1, P p2, P q1, P q2)
{
    double a1 = cross(q1, q2, p1), a2 = -cross(q1, q2, p2);
    return sign(a1 + a2) != 0;
}
//直线交点
P isll(P p1, P p2, P q1, P q2)
{
    double a1 = cross(q1, q2, p1), a2 = -cross(q1, q2, p2);
    return (p1 * a2 + p2 * a1) / (a1 + a2);
}

int main()
{
}