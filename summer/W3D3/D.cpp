#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 1000;
int tot1, tot2;
int mp[N][N];
int link[N];
int vis[N];
int dfs(int u)
{
    int v;
    for (v = 0; v < tot2; v++)
        if (mp[u][v] && !vis[v])
        {
            vis[v] = 1;
            if (link[v] == -1 || dfs(link[v]))
            {
                link[v] = u;
                return 1;
            }
        }
    return 0;
}
int n, m, k, board[33][33];
int toi[4] = {0, 0, 1, -1};
int toj[4] = {1, -1, 0, 0};
int main()
{
    while (scanf("%d%d%d", &n, &m, &k) != EOF)
    {
        memset(mp, 0, sizeof(mp));
        memset(board, 0, sizeof(board));
        tot1 = 0;
        tot2 = 0;
        for (int i = 1; i <= k; i++)
        {
            int x, y;
            scanf("%d%d", &x, &y);
            board[y][x] = -1;
        }
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
            {
                if (board[i][j] == -1)
                    continue;
                if ((i + j) % 2)
                    board[i][j] = tot1++;
                else
                    board[i][j] = tot2++;
            }
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                if (board[i][j] != -1 && ((i + j) % 2))
                {
                    for (int k = 0; k <= 3; k++)
                    {
                        int nexti = i + toi[k];
                        int nextj = j + toj[k];
                        if (nexti > n || nexti < 1 || nextj > m || nextj < 1 || board[nexti][nextj] == -1)
                            continue;
                        mp[board[i][j]][board[nexti][nextj]] = 1;
                    }
                }
        int res=0;
        memset(link, -1, sizeof(link));
        for (int i = 0; i < tot1; i++)
        {
            memset(vis, 0, sizeof(vis));
            if (dfs(i))
                res++;
        }
        if (res * 2 == m * n - k)
            printf("YES\n");
        else
            printf("NO\n");
    }
}