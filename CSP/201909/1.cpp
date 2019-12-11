#include <cstdio>
#include <iostream>
using namespace std;
int n, m;
int tot, num, max_num;
int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int tmp, a, yuan;
        cin >> tmp;
        yuan = tmp;
        for (int j = 1; j <= m; j++)
        {
            cin >> a;
            tmp += a;
        }
        tot += tmp;
        if (max_num < yuan - tmp)
        {
            max_num = yuan - tmp;
            num = i;
        }
    }
    cout << tot << " " << num << " " << max_num << endl;
}
/*
3 3
73 -8 -6 -4
76 -5 -10 -8
80 -6 -15 0
*/