#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
using namespace std;
string space = "\\x20", reset = "\\x1B\\x5B\\x30\\x6D";
int m, n, p, q; //m 列 n 行 p 列 q 行
int block;
int hex2dec(char a1, char a2)
{
    int re = 0;
    if (isdigit(a1))
        re += 16 * (a1 - '0');
    else
        re += 16 * (a1 - 'a' + 10);
    if (isdigit(a2))
        re += a2 - '0';
    else
        re += a2 - 'a' + 10;
    return re;
}
struct COLOR
{
    int r, g, b;
    COLOR const operator+(const COLOR &o)
    {
        COLOR re;
        re.r = this->r + o.r;
        re.g = this->g + o.g;
        re.b = this->b + o.b;
        return re;
    }
    COLOR const operator/(const int &o)
    {
        COLOR re;
        re.r = this->r / o;
        re.g = this->g / o;
        re.b = this->b / o;
        return re;
    }
    bool const operator==(const COLOR &o)
    {
        if (this->r == o.r && this->g == o.g && this->b == o.b)
            return true;
        else
            return false;
    }
    void read()
    {
        string raw, color;
        cin >> raw;
        if (raw.size() == 2)
        {
            for (int i = 1; i <= 6; i++)
            {
                color += raw[1];
            }
        }
        else if (raw.size() == 4)
        {
            color += raw[1];
            color += raw[1];
            color += raw[2];
            color += raw[2];
            color += raw[3];
            color += raw[3];
        }
        else
        {
            for (int i = 1; i <= 6; i++)
            {
                color += raw[i];
            }
        }
        r = hex2dec(color[0], color[1]);
        g = hex2dec(color[2], color[3]);
        b = hex2dec(color[4], color[5]);
    }
    void print_num(int a)
    {
        int flag = 0;
        if (a / 100 > 0)
        {
            printf("\\x3%d", a / 100);
            a %= 100;
            flag = 1;
        }
        if (a / 10 > 0 || flag)
        {
            printf("\\x3%d", a / 10);
            a %= 10;
        }
        printf("\\x3%d", a);
    }
    void print_color()
    {
        printf("\\x1B\\x5B\\x34\\x38\\x3B\\x32\\x3B");
        //r;g;b
        print_num(this->r);
        printf("\\x3B");
        print_num(this->g);
        printf("\\x3B");
        print_num(this->b);
        printf("\\x6D\\x20");
    }
};
COLOR raw[1080][1920], pro[1080][1920];
COLOR black;
int main()
{
//    freopen("1.txt", "r", stdin);
    freopen("3.out", "w", stdout);
    scanf("%d %d %d %d", &m, &n, &p, &q);
    block = (m * n) / (p * q);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            raw[i][j].read();
        }
    }
    for (int i = 0; i < n; i += q)
    {
        for (int j = 0; j < m; j += p)
        {
            for (int k = 0; k < q; k++)
            {
                for (int l = 0; l < p; l++)
                {
                    pro[i / q][j / p] = pro[i / q][j / p] + raw[i + k][j + l];
                }
            }
            pro[i / q][j / p] = pro[i / q][j / p] / (p * q);
        }
    }
    black.r = 0;
    black.g = 0;
    black.b = 0;
    COLOR now = black;
    for (int i = 0; i < n / q; i++)
    {
        for (int j = 0; j < m / p; j++)
        {
            if (pro[i][j] == now)
            {
                cout << space;
            }
            else if (pro[i][j] == black)
            {
                now = black;
                cout << reset << space;
            }
            else
            {
                now = pro[i][j];
                pro[i][j].print_color();
            }
        }
        if (!(now == black))
        {
            now = black;
            cout << reset;
        }
        cout << "\\x0A";
    }
}